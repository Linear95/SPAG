# Data Information

- [`all_target_words.txt`](https://drive.google.com/file/d/1M7574n9EHIXkkrZjBX9X0S2RTSHKfmGN/view?usp=drive_link) contains all the target words to play the adversarial games.

- [`gpt4_game_top30k_results.json`](https://drive.google.com/file/d/1xQ5E0UHtdiY8Kp4I_e3umdL0RJrOS3Se/view?usp=drive_link) collects the self-play episodes of GPT-4 on the top-30K words in `all_target_words.txt`.

- [`train_imitation_gpt4.json`](https://drive.google.com/file/d/1iqm4ZuZ_uMm0DaEZt_0ho4fv9WM4_tOo/view?usp=drive_link) converts `gpt4_game_top30k_results.json` into the instruction-tuning format, with different prompt templates randomly selected from `GAME_RULE_PROMPTS` in `utils.py`.


