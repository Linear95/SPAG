import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import random

import torch
import torch.distributed as dist
import transformers

from torch.utils.data import Dataset
from transformers import Trainer, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM


from dataloaders import TextDataset
from dataloaders import sft_data_collactor, offline_ppo_data_collactor, weighted_sft_data_collactor
from arguments import CustomTrainingArguments

from trainers import SFTWeightedWithKLTrainer, OfflineWeightedPolicyTrainer

from utils import print_rank_0, read_json_or_jsonl_data
from utils import DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN


def get_train_dataset(args):    
    all_train_data = []
    for train_data_path in args.train_data_path:
        train_data = read_json_or_jsonl_data(train_data_path)
        all_train_data.extend(train_data)

    if args.debug_mode:
        print_rank_0(f">>> check loaded data:")        
        print_rank_0(f">>> {all_train_data[0]}")

    train_set = TextDataset(all_train_data)
    return train_set


def train():
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print_rank_0(args)

    # load data
    #---------------------------------------------------------------------------------
    train_dataset = get_train_dataset(args)    

    # setup model
    #---------------------------------------------------------------------------------
    print_rank_0(f"Begin loading model from {args.model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,            
    )
    if hasattr(model, "ref_model"):
        del model.ref_model

    if args.train_method in ["sft_weighted_with_kl", "offlinePO"] \
      and args.ref_model_name_or_path:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_name_or_path,
            trust_remote_code=True,
        )
        if hasattr(ref_model, "ref_model"):
            del ref_model.ref_model
        for param in ref_model.parameters():
            param.requires_grad = False

        model.ref_model = ref_model
        
    print_rank_0(model)
    print_rank_0(f"Finished loading model from {args.model_name_or_path}")

    model.is_parallelizable = True
    model.model_parallel = True

    # setup tokenizer
    #---------------------------------------------------------------------------------        
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,      
        model_max_length=args.max_length,        
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        use_fast=True,
        trust_remote_code=True,

    )

    if tokenizer.pad_token is None:
        # We do not resize the vocab embedding, since it ruins the KL value with the ref_model
        tokenizer.pad_token_id = 0 
        tokenizer.pad_token = tokenizer.decode(0)
        print_rank_0("set pad token id to 0")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id


    print_rank_0(tokenizer)

    # build trainer
    #---------------------------------------------------------------------------------

    if args.train_method == "sft":
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer, 
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: sft_data_collactor(args, x, tokenizer)
        )
        
    elif args.train_method == "sft_weighted_with_kl":
        trainer = SFTWeightedWithKLTrainer(
            model=model,
            tokenizer=tokenizer, 
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: offline_ppo_data_collactor(args, x, tokenizer)
        )
        
    elif args.train_method == "offlinePO":
        trainer = OfflineWeightedPolicyTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda x: offline_ppo_data_collactor(args, x , tokenizer)
        )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    trainer.save_state()
    if hasattr(trainer.model, "ref_model"):
        del trainer.model.ref_model


    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    train()
