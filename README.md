# Self-Play of Adversarial Language Game (SPAG)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/Linear95/SPAG/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/Linear95/SPAG/blob/main/DATA_LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This repo contains the implementation of the paper:

- [Self-playing Adversarial Language Game Enhances LLM Reasoning](https://arxiv.org/abs/2404.10642)

We explore the **S**elf-**P**lay training of LLMs
in an **A**dversarial language **G**ame (SPAG) named [*Adversarial Taboo*](https://arxiv.org/abs/1911.01622).

<p align="center">
  <img src="figures/spag-reasoning-plot.png" height="80%" width="80%">
  <figcaption> Examples of Adversarial Taboo with the target word "conversation". Left is an attacker-winning game, while right is a defender-winning game.  </figcaption>
</p>


With the training epoch of SPAG increasing, the LLM reasoning ability continuously improves as shown in plots below:
<p align="center">
  <img src="figures/game_examples.png" height="80%" width="80%">
</p>


## Environment
To build the running environment, use the following command:
```
pip3 install -r requirements.txt
```

## Imitation Learning
To ensure the instruction-following ability of LLMs to the game rules, we first let LLMs imitate the winning behaviors of GPT-4.
To launch the imitation learning on LLaMA-2-7B-base, use the following command:
```bash
torchrun --nproc_per_node=8 --master_port=6000 train.py \
    --output_dir <path_to_save_your_imitation_checkpoint> \
    --model_name_or_path "Llama-2-7b-hf" \
    --ref_model_name_or_path "Llama-2-7b-hf" \
    --lm_kl_coeff 0.1 \
    --train_method "SFTwithKL" \
    --train_data_path "./data/train_imitation_gpt4.json" \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy no \
    --padding_side "right" \
    --truncation_side "left" \
    --max_length 2048 \
    --save_strategy epoch \
    --learning_rate 5e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --weight_decay 0. \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --gradient_checkpointing True \
    --tf32 True  --bf16 True
```

Here [`Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) can be replaced by [`Baichuan2-13B-Base`](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base) to reproduce the Baichuan-2 results in our paper.


## Self-play Episode Collection

After the imitation learning, we can conduct the self-play with the imitation-learned model on all targets words:

```bash
export PYTHONPATH=.

torchrun --nproc_per_node=8 --master_port=6000 tools/play_llm_game.py \
    --taboo_max_turns 5 \
    --attacker_model_name_or_path <path_to_imitation_learned_model> \
    --defender_model_name_or_path <path_to_imitation_learned_model> \
    --model_prefix "im_llama2" \
    --data_path "./data/all_target_words.txt" \
    --output_dir "./data/self_play_results" \
    --per_device_eval_batch_size 1 \
    --task_type "sampling" \
    --data_suffix "all_words" \
    --max_length 2048 \
    --max_new_tokens 256 \
    --logging_steps 5 \
    --bf16 True  --tf32 True
```

When the self-play collection is finished, we can access all the game episodes in `im_llama2_sampling_all_words_results.json` at `data/self_play_results/`.


## Reinforcement Learning on Self-play Episodes
To conduct reinforcement learning on game episodes, we first calculate the outcomes by rule-based judgment and assign rewards to actions:
```bash
export PYTHONPATH=.

python3 tools/assign_rewards.py \
    --input_data_path data/self_play_results/im_llama2_sampling_all_target_words_results.json \
    --output_data_path data/train_spag_data_im_llama2.json \
    --sft_data_path data/alpaca_train.json
```

The output file `train_spag_data_im_llama2.json` is already in an instruction-tuning format, with following keywords:

- `query` \& `target`: the input and label for language modeling,
- `reward`: the reward assigned to current utterance (`target`),
- `weight`: the re-weighting paramenter to ensure that both attacker and defender have equal learning coefficient 1/2 in expectation.

Then the SPAG model can be learned with the following command:
```bash
torchrun --nproc_per_node=8 --master_port=6000 train.py \
    --output_dir <path_to_save_your_SPAG_checkpoint> \
    --model_name_or_path <path_to_your_imitation_checkpoint> \
    --ref_model_name_or_path <path_to_your_imitation_checkpoint> \
    --lm_kl_coeff 0.2 \
    --lm_sft_coeff 0.5 \
    --train_method "OfflinePO" \
    --train_data_path "./data/train_spag_data_im_llama2.json" \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy no \
    --padding_side "right" \
    --truncation_side "left" \
    --max_length 2048 \
    --save_strategy epoch \
    --learning_rate 2e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --weight_decay 0. \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --gradient_checkpointing True \
    --tf32 True  --bf16 True
```
By repeating the episode-collection and SPAG-learning processes, we can observe continous improvements on reasoning benchmarks. For LLM reasoning evaluation, we use the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repo with the setups described in our paper.

## Citation
Please cite our paper if you find the code useful.
```
@article{cheng2024spag,
  title={Self-playing Adversarial Language Game Enhances LLM Reasoning},
  author={Cheng, Pengyu and Hu, Tianhao and Xu, Han and Zhang, Zhisong and Dai, Yong and Han, Lei and Du, Nan},
  journal={arXiv preprint arXiv:2404.10642},
  year={2024}
}
```
