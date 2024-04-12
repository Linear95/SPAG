import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import datetime

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

import transformers
from transformers import Trainer, AutoConfig

from utils import print_rank_0, IGNORE_INDEX


def compute_lm_loglikeli(logits, labels):
    batch_size, seq_length, vocab_size = logits.shape
        
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels).reshape(batch_size, -1) # [bs * seq_len]
    ignore_mask = labels != IGNORE_INDEX
    
    mean_loss = loss.sum(dim=-1) / ignore_mask.sum(dim=-1)

    return - mean_loss #loss.reshape(batch_size, -1).mean(dim=-1)


class SFTWeightedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.debug_mode:
            print_rank_0(f"check inputs :{inputs}")
            
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        logits = model_outputs.logits # [bs, seq_len, vocab]

        batch_size, seq_length, vocab_size = logits.shape
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['labels'][..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels) # [bs * seq_len]
        weighted_loss = (loss.reshape(batch_size, -1).mean(dim=1) * inputs['weights']).mean() # [batch_size]

        if self.args.debug_mode:
            print_rank_0(f"check logits : {logits}")
            print_rank_0(f"check loss : {loss}")
            print_rank_0(f"check weighted loss : {weighted_loss}")

        return (weighted_loss, outputs) if return_outputs else weighted_loss
        
        
class SFTWeightedWithKLTrainer(Trainer):


    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.debug_mode:
            print_rank_0(f"check inputs :{inputs}")
            
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        with torch.no_grad():
            # model.ref_model.eval()
            ref_model_outputs = model.ref_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            ref_logprob = compute_lm_loglikeli(ref_model_outputs.logits, inputs['labels']) #[batch_size]

        if self.args.debug_mode:
            print_rank_0(f"check ref_model output: {ref_logprob}")

        logprob = compute_lm_loglikeli(model_outputs.logits, inputs['labels'])

        # for importance sampling
        # loss = - logprob + self.args.llm_kl_coeff * (logprob - ref_logprob)

        # for MC kl
        kl_divergence = logprob.exp() * (logprob - ref_logprob)

        if self.args.use_kl_mask:
            lm_size, kl_size = (1-inputs['sft_mask']).sum(), (inputs['sft_mask']).sum()
            
            lm_loss = (-logprob * (1 - inputs['sft_mask']) * inputs['weights']).sum() / lm_size if lm_size > 0 else lm_size
            kl_loss = (inputs['sft_mask'] * kl_divergence).sum() / kl_size if kl_size > 0 else kl_size

            total_loss = lm_loss + self.args.lm_kl_coeff * kl_loss
        else:        
            loss = - logprob + self.args.lm_kl_coeff * kl_divergence

        #total_loss = - logprob * self.args.lm_loss_coeff * inputs['sft_mask'] + ppo_loss * (1 - inputs['sft_mask'])
        
            total_loss = (loss * inputs['weights']).mean() # [batch_size]

        # total_loss = weighted_loss + args.llm_kl_coeff * kl_divergence

        if self.args.debug_mode:          
            print_rank_0(f"check loss : {loss}")
            print_rank_0(f"check weighted loss : {weighted_loss}")
            print_rank_0(f"check kl divergence : {kl_divergence}")

        return (total_loss, outputs) if return_outputs else total_loss



class OfflineWeightedPolicyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.debug_mode:
            print_rank_0(f"check inputs :{inputs}")
            
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        with torch.no_grad():
            # model.ref_model.eval()
            ref_model_outputs = model.ref_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            ref_logprob = compute_lm_loglikeli(ref_model_outputs.logits, inputs['labels']).detach() #[batch_size]

        if self.args.debug_mode:
            print_rank_0(f"check ref_model output: {ref_logprob}")

        logprob = compute_lm_loglikeli(model_outputs.logits, inputs['labels'])

        # for importance sampling
        # loss = - logprob + self.args.llm_kl_coeff * (logprob - ref_logprob)

        # ratio clipped method
        # ===============================
        if self.args.use_kl_mask:
            kl_div = logprob.exp() * (logprob - ref_logprob)
        else:
            kl_div = (logprob - ref_logprob)            
            
        # importance_ratio = torch.clip((logprob - ref_logprob).exp(), 1 - self.args.clip_range, 1 + self.args.clip_range)        
        importance_ratio = (logprob - ref_logprob).exp()
        importance_ratio_clipped = torch.clip(importance_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)

        if self.args.use_kl_mask:
            advantages = inputs['rewards'] 
        else:
            advantages = inputs['rewards'] - self.args.lm_kl_coeff * kl_div

        ppo_loss = - torch.minimum(advantages * importance_ratio, advantages * importance_ratio_clipped)

        if self.args.use_kl_mask:
            total_loss = self.args.lm_kl_coeff * kl_div * inputs['sft_mask'] + ppo_loss * (1 - inputs['sft_mask'])
        else:
            sample_size, sft_size = (1-inputs['sft_mask']).sum(), (inputs['sft_mask']).sum()
            sft_loss = (- logprob * inputs['sft_mask']).sum() / sft_size if sft_size > 0 else sft_size
            ppo_loss = (ppo_loss * (1 - inputs['sft_mask'])).sum() / sample_size if sample_size > 0 else sample_size
            #total_loss = - logprob * self.args.lm_loss_coeff * inputs['sft_mask'] + ppo_loss * (1 - inputs['sft_mask'])
            total_loss = self.args.lm_loss_coeff * sft_loss + ppo_loss
        
        # loss = - logprob * inputs['rewards'] * importance_ratio.detach()  # self.args.lm_kl_coeff * kl_div
        
        weighted_loss = (total_loss * inputs['weights']).mean() # [batch_size]

        if self.args.debug_mode:          
            print_rank_0(f"check loss : {loss}")
            print_rank_0(f"check weighted loss : {weighted_loss}")
            print_rank_0(f"check kl divergence : {kl_div}")

        return (weighted_loss, outputs) if return_outputs else weighted_loss
