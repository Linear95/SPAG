# Data Information

- [`all_target_words.txt`](https://drive.google.com/file/d/1M7574n9EHIXkkrZjBX9X0S2RTSHKfmGN/view?usp=drive_link) contains all the target words to play the adversarial games.

- [`gpt4_game_top30k_results.json`](https://drive.google.com/file/d/1xQ5E0UHtdiY8Kp4I_e3umdL0RJrOS3Se/view?usp=drive_link) collects the self-play episodes of GPT-4 on the top-30K words in `all_target_words.txt`.

- [`train_imitation_gpt4.json`](https://drive.google.com/file/d/1iqm4ZuZ_uMm0DaEZt_0ho4fv9WM4_tOo/view?usp=drive_link) converts `gpt4_game_top30k_results.json` into the instruction-tuning format, with different prompt templates randomly selected from `GAME_RULE_PROMPTS` in `utils.py`.

- [`alpaca_train.json`](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) is the SFT data from the [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) repo. Note that you need to use the prompt [templates](https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L31) to convert `'instruction'` and `'input'` to `'prompt'`, and `'output'` to `'target'`. 

- [`im_llama2_sampling_all_words_results.json`](https://drive.google.com/file/d/12QtjUb3hirajM2e5zfgnSduETmPnvb80/view?usp=drive_link) includes our inference about all game episodes of the LLaMA-2 imitation-learned model.

- [`SPAG1_sampling_all_words_results.json`](https://drive.google.com/file/d/1nvJp_f6M9Mg7bcGnccUgITyW7nZ1hQ2C/view?usp=drive_link) includes our inference about all game episodes of the LLaMA-2 SPAG-1 model.

- [`SPAG2_sampling_all_words_results.json`](https://drive.google.com/file/d/13Tf3PC5ioRzSTGCobMZWmx1uFWi4jS1F/view?usp=drive_link) includes our inference about all game episodes of the LLaMA-2 SPAG-2 model.
