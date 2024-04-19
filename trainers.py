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
    
    avg_loss = loss.sum(dim=-1) / ignore_mask.sum(dim=-1)

    return - avg_loss

        
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

        # for MC kl
        kl_divergence = logprob.exp() * (logprob - ref_logprob)

        loss = - logprob + self.args.lm_kl_coeff * kl_divergence        
        
        total_loss = (loss * inputs['weights']).mean() # [batch_size]

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
        kl_div = (logprob - ref_logprob)
        
        importance_ratio = (logprob - ref_logprob).exp()
        importance_ratio_clipped = torch.clip(importance_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)

        advantages = inputs['rewards'] - self.args.lm_kl_coeff * kl_div
        ppo_loss = - torch.minimum(advantages * importance_ratio, advantages * importance_ratio_clipped)

        sample_size, sft_size = (1-inputs['sft_mask']).sum(), (inputs['sft_mask']).sum()
        sft_loss = (- logprob * inputs['sft_mask']).sum() / sft_size if sft_size > 0 else sft_size
        ppo_loss = (ppo_loss * (1 - inputs['sft_mask'])).sum() / sample_size if sample_size > 0 else sample_size
        
        total_loss = self.args.lm_sft_coeff * sft_loss + ppo_loss                
        
        weighted_loss = (total_loss * inputs['weights']).mean() # [batch_size]

        if self.args.debug_mode:          
            print_rank_0(f"check loss : {loss}")
            print_rank_0(f"check weighted loss : {weighted_loss}")
            print_rank_0(f"check kl divergence : {kl_div}")

        return (weighted_loss, outputs) if return_outputs else weighted_loss
