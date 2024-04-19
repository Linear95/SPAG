import re
import json
from typing import List
from tqdm import tqdm
import argparse

# from textblob import TextBlob

from utils import randomly_convert_game_history_to_query

PREDICT_TEMP = r"i know the word! it.{1,8}"

def get_derivative_words(word: str):
    # fuzzy matching for similar words 
    word = word.lower()
    # blob_word = TextBlob(word)
    word_list = [word,  word+'ing', word+'ed', #blob_word.words.pluralize()[0],
                 f"`{word}`", f'"{word}"', f"'{word}'", f"\"{word}\""]
    
    return word_list


def has_target_word(content: str, target_word: str):
    derivative_words = get_derivative_words(target_word)
    return any([word in content for word in derivative_words])


def is_prediction(content: str, target_word: str):

    if re.search(PREDICT_TEMP, content):
        return True
    else:
        return False

def is_correct_prediction(content: str, target_word: str):
    derivative_words = get_derivative_words(target_word)
    predict_regex = [PREDICT_TEMP + word for word in derivative_words]

    if any([re.search(temp, content) for temp in predict_regex]):
        return True
    else:
        return False
    

def get_game_outcome(history, target_word, max_turns):
    history_length = 0
    for i, item in enumerate(history):
        history_length += 1
        if item['role'] == 'defender':
            if is_prediction(item['content'], target_word):
                if is_correct_prediction(item['content'], target_word):
                    return "defender wins", history_length
                else:
                    return "attacker wins", history_length
                
            elif has_target_word(item['content'], target_word):
                return "attacker wins", history_length
                
        else:
            if has_target_word(item['content'], target_word):
                return 'attacker breaks the rules', history_length
            elif is_prediction(item['content'], target_word):
                return 'attacker breaks the rules', history_length

        if history_length >= max_turns * 2:
            break

    return "tied game", history_length



def compute_self_play_sample_rewards(game_episodes, decay_weight=0.8):
    defender_game_num, attacker_game_num = 0, 0
    increase_weight = 1 / decay_weight
    outputs = []
    for item in game_episodes:
        outcome, history_length = get_game_outcome(item['history'], item['target_word'], item['max_turns'])

        if outcome == "attacker wins":
            attacker_game_num += 1                       
            winner = "attacker"
            
        elif outcome == "defender wins":
            defender_game_num += 1                       
            winner = "defender"
        
        else:
            continue
                            
        reward_coeff = 1.
        current_turn = 0.
        total_reward = 1.
        total_reward_coeff = 0.
        
        for i, message in enumerate(item['history'][:history_length]):
            if message['role'] != winner:
                continue
                
            query = randomly_convert_game_history_to_query(
                item['history'][:i],
                target_word=item['target_word'],
                max_turns=item['max_turns']
            )
            
            target = message['content']

            current_turn += 1
            reward_coeff *= increase_weight
            total_reward_coeff += reward_coeff

            message['query'] = query
            message['target'] = target
            message['reward_coeff'] = reward_coeff

        for i, message in enumerate(item['history'][:history_length]):
            if 'query' in message:
                outputs.append({
                    "query": message['query'],
                    "target": message['target'].strip(),
                    "reward": total_reward * message['reward_coeff'] / total_reward_coeff,
                    "role": message['role']
            })


    all_game_num = attacker_game_num + defender_game_num

    # to ensure that both attacker and defender have learning coefficient 1/2 in expectation.
    defender_weight = all_game_num / (2 * defender_game_num) if defender_game_num > 0 else 0.
    attacker_weight = all_game_num / (2 * attacker_game_num) if attacker_game_num > 0 else 0.
    
    print(f"totally get {len(outputs)} data from {all_game_num} game, with {attacker_game_num} attacker games;  {defender_game_num} defender games.")
    
    print("reweight the sample with attacker_weight: {} ; defender_weight: {}".format(attacker_weight, defender_weight))

    for item in outputs:
        if item['role'] == "attacker":
            item['weight'] = attacker_weight  
        else:
            item['weight'] = defender_weight
    return outputs

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='parser for episode processing.')
    parser.add_argument("--input_data_path", type=str, default="", help="the path to input data.")
    parser.add_argument("--output_data_path", type=str, default="", help="the path to output data.")
    parser.add_argument("--sft_data_path", type=str, default="", help="the path to load sft data.")    
    parser.add_argument("--decay_weight", type=float, default=0.8, help="the decay weight of reward.")

    args = parser.parse_args()
    
    with open(args.input_data_path, 'r') as f:
        game_episodes = json.load(f)

    results = compute_self_play_sample_rewards(game_episodes, args.decay_weight)

    if args.sft_data_path:
        with open(args.sft_data_path, 'r') as f:
            sft_data = json.load(f)
    else:
        sft_data = []

    for item in sft_data:
        item['type'] = 'sft'
        item['weight'] = 1.

    with open(args.output_data_path, 'w') as f:
        json.dump(results + sft_data, f, ensure_ascii=False, indent=2)
