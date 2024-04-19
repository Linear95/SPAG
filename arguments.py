from typing import List, Optional, Tuple, Union

from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class CustomTrainingArguments(TrainingArguments):

    # tokenizer params
    padding_side: str = field(
        default="right",
        metadata={"help": "the direction for tokenizer to add padding tokens."}
    )

    truncation_side: str = field(
        default="left",
        metadata={"help": "the direction for tokenizer to add padding tokens."}
    )

    add_sep_token: bool =field(
        default=False,
        metadata={"help": "whether add a <sep> token between query and response."}
    )

    tokenizer_path: str = field(
        default="llama-7b-hf", 
        metadata={"help": "the path to load pretrained tokenizer."}
    )

    resize_vocab:  bool =field(
        default=False,
        metadata={"help": "whether resize the vocabulary to add special pad token for llama."}
    )


    # model params
    model_type: str = field(
        default="llama",
        metadata={"help": "the base model type for reward model, selected from [llama, bert]."}
    )

    model_prefix: str = field(
        default="llama",
        metadata={"help": "the base model type for reward model, selected from [llama, bert]."}
    )


    pooling_type: str = field(
        default="average",
        metadata={"help": "the pooling method for reward model, selected from [average, max, last]."}
    )

    model_name_or_path: str = field(
        default="llama-7b-hf", 
        metadata={"help": "the path to load pretrained model."}
    )

    ref_model_name_or_path: str = field(
        default="", 
        metadata={"help": "the path to load reference model."}
    )

    attacker_model_name_or_path: str = field(
        default="", 
        metadata={"help": "the path to load reference model."}
    )

    defender_model_name_or_path: str = field(
        default="", 
        metadata={"help": "the path to load reference model."}
    )


    # data params
    taboo_max_turns: int = field(
        default=5,
        metadata={"help": "the max_turn to play the adversarial taboo game."}
    )

    data_dir: str = field(
        default="path/to/cleaned_data",
        metadata={"help": "the directory to load data."}
    )   

    data_type: str = field(
        default="no_type",
        metadata={"help": "the type of data."}
    )
    data_path: str = field(
        default="yahma/alpaca-cleaned",
        metadata={"help": "the path to load data."}
    )   

    train_data_path: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "train datasets paths."}
    )


    eval_data_path: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "evaluation datasets paths."}
    )

    data_prefix: str = field(
        default="yahma/alpaca-cleaned",
        metadata={"help": "the prefix to load train and test data."}
    )

    data_suffix: str = field(
        default="yahma/alpaca-cleaned",
        metadata={"help": "the suffix to save inference data."}
    )   

    # training hyperparams
    task_type: str = field(
        default="training",
        metadata={"help": "the task type"}
    )

    train_method: str = field(
        default="sft",
        metadata={"help": "the LLM training method name."}
    )

    
    debug_mode: bool = field(
        default=False,
        metadata={"help": "whether use the debug mode."}
    )

    cache_dir: Optional[str] = field(default=None)

    optim: str = field(default="adamw_torch", metadata={"help": "the paramter to use"})

    clip_range: float = field(default=0.2, metadata={"help": "the range to clip the importance reweighting ratio for policy optimization."})

    length_penalty: float = field(default=1., metadata={"help": "the penalty for seq length."})

    lm_sft_coeff: float = field(default=0., metadata={"help": "the coefficient for SFT data language modeling loss."})            

    lm_kl_coeff: float = field(default=0., metadata={"help": "the coefficient of kl regularizer."})

    max_length: int = field(
        default=256,
        metadata={"help": "the max sentence sequence length."}
    )   

    batch_size: int = field(
        default=256,
        metadata={"help": "the overall training batch size"}
    )   

    micro_batch_size: int = field(
        default=32,
        metadata={"help": "the batch size on each device, equavilent to `per_gpu_train_batch_size`"}
    )


    valid_data_size: int = field(
        default=0,
        metadata={"help": "the data size for validation data"}
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None, 
        metadata={"help":  "either training checkpoint or final adapter"}
    )
    # generation parameters:
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "the max sentence sequence length."}
    )   
