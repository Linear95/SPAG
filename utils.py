import math
import os
import io
import sys
import time
import json
import glob
import random
from typing import Optional, Sequence, Union, List, Dict

import tqdm
import copy

import numpy as np
import torch

import transformers

IGNORE_INDEX=-100

SEP_TOKEN="<sep>"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

GAME_RULE_PROMPTS=[
'''Play the game of Adversarial Taboo. In this game, there are two players, an attacker and a defender.

At the begining, attacker is assigned with a target word, with which the defender is not informed. The task of the attacker is to induce the defender to utter the target word unconciously, then the attacker win the game. However, the attacker is not allow to utter the target word, otherwise the attacker loses.

At the same time, the defender tries to figure out the target word. if the defender indentifies the target word, he can speak "I know the word! It is `target word`!". Then the defender wins. Remember, the defender can only guess the word once, otherwise he will directly lose the game.

The game has {max_turns} turns. if nobody wins during {max_turns} turns, there is a tie.
''',
'''
Engage in the strategic contest of Adversarial Taboo, a game involving two participants: one assumes the role of the attacker, while the other is the defender.

Initially, the attacker is secretly given a target word, which remains unknown to the defender. The attacker's objective is to cleverly prompt the defender to say the target word without realizing it. If the defender does so, the attacker triumphs. However, the attacker must refrain from saying the target word themselves; doing so results in an immediate loss.

Concurrently, the defender's goal is to deduce the target word. Upon successful identification, the defender may confidently declare, "I know the word! It is 'target word'!" If correct, the defender claims victory. It's crucial to note that the defender is granted only one opportunity to guess the word correctly; any additional guesses lead to an automatic failure.

The game unfolds over {max_turns} rounds. Should neither player succeed within these turns, the game ends in a draw.
''',
'''
Dive into the cunning duel known as Adversarial Taboo, where two contenders face off: one as the attacker, the other as the defender.

To kick things off, the attacker is covertly assigned a target word, which is kept secret from the defender. The attacker's mission is to subtly coax the defender into saying this word without their awareness. Success means victory for the attacker. But there's a catch: if the attacker slips up and says the word themselves, they lose.

Meanwhile, the defender is on a quest to uncover the target word. Should the defender succeed and exclaim, "I know the word! It is 'target word'!" then victory is theirs. Caution is key for the defender, who is allowed only a single guess at the word; any more and they automatically lose.

This mind game is played in {max_turns} rounds. If at the end of these rounds no one has won, the match is declared a tie.
''',
'''
Step into the challenge of Adversarial Taboo, a game for two: one as the attacker, the other as the defender.

In the beginning, a target word is secretly given to the attacker, unknown to the defender. The attacker's challenge is to lead the defender to say the target word without their knowledge, securing a win for the attacker. But there's a twist: the attacker must avoid saying the target word themselves, or they forfeit the game.

Simultaneously, the defender is on a mission to guess the target word. If the defender figures it out, they can announce, "I know the word! It is 'target word'!" and if they're right, they win. However, the defender must tread carefully, as they have only one chance to guess correctly; a wrong guess means instant defeat.

The game proceeds over {max_turns} rounds. If neither player prevails within these rounds, the game ends in a stalemate.
''',
'''
Immerse yourself in the strategic face-off called Adversarial Taboo, where two roles are in play: an attacker and a defender.

As the game sets in motion, the attacker is discreetly handed a target word, which remains a secret to the defender. The attacker's goal is to subtly manipulate the defender into saying this specific word without their awareness, which would result in a win for the attacker. But there's a rule: the attacker must never speak the target word themselves, or they will be defeated.

Concurrently, the defender is engaged in a mental game of detection, aiming to identify the target word. If the defender manages to pinpoint the word, they can declare, "I know the word! It is 'target word'!" A correct identification means the defender wins. It's important to note that the defender gets only one shot at guessing the word; any incorrect guess leads to an immediate loss.

The game unfolds across {max_turns} rounds. If by the end of these turns no one has emerged victorious, the game is considered a draw.
''',
'''
Dive into the strategic duel known as Adversarial Taboo, where two roles emerge: a defender and an attacker.

The defender starts off in the dark, with the attacker being secretly assigned a target word. The defender's mission is to deduce this word. If successful, they can proclaim, "I know the word! It is 'target word'!" and claim victory. But caution is key — the defender is allowed only one guess. A wrong guess results in an immediate loss.

Meanwhile, the attacker's aim is to covertly lead the defender to say the target word without their knowledge. If the defender speaks the word unwittingly, the attacker triumphs. However, if the attacker slips and says the target word themselves, they lose.

The game is played over {max_turns} rounds. If no winner is determined after these rounds, the game ends in a tie.
''',
'''
Step into the intriguing game known as Adversarial Taboo, where the roles of defender and attacker are pivotal.

The defender embarks on a cerebral journey, unaware of the secret target word that the attacker has been given. The defender's challenge is to uncover this word. A correct declaration of "I know the word! It is 'target word'!" secures a win for the defender. However, they must tread carefully, as they have only one chance to guess correctly; any incorrect guess spells instant defeat.

On the other side of the gameboard, the attacker is tasked with a delicate mission: to nudge the defender into saying the target word without their conscious realization. Success in this stealthy endeavor means victory for the attacker. But there's a catch: if the attacker accidentally mentions the target word, they lose the game.

The tension builds over four rounds. If by the end of these rounds the game has not been won, it is declared a stalemate.
''',
'''
Embark on the strategic challenge of Adversarial Taboo, where players assume the roles of either attacker or defender.

The defender enters the game blind to the secret target word that only the attacker knows. The goal for the defender is to guess this word. If they can confidently assert, "I know the word! It is 'target word'!" they win. But they must be cautious, as they have only one opportunity to make a correct guess; a wrong guess means they lose.

Conversely, the attacker's goal is to subtly coax the defender into saying the target word without them being aware of it. If the defender says the word, the attacker wins. However, should the attacker accidentally say the target word themselves, they forfeit the game.

The game unfolds over the course of {max_turns} rounds. If neither player has achieved victory by the end of the {max_turns} round, the game ends in a draw.
'''
]


INSTRUCT_PROMPTS = {
    "attacker": '''\n\n### Instruction: You are the attacker. The target word is `{target_word}`. Provide your response for the next turn.\n\n### Response:''', 
    "defender": '''\n\n### Instruction: Your are the defender. Provide your response to infer the target word.\n\n### Response:'''
}

PLAYER_INSTRUCT_PROMPTS = {
    "attacker": "You are the attacker. The target word is `{target_word}`. Provide your response for the next turn.", 
    "defender": "Your are the defender. Provide your response to infer the target word."
}


def convert_game_history_to_query(history, target_word, max_turns=5):
    GAME_RULE_PROMPT = GAME_RULE_PROMPTS[0]
    history_str = ""
    for i, message in enumerate(history):
        history_str += "\n  - {}: {}".format(message['role'], message['content'])
    
    if len(history) == 0:
        query = GAME_RULE_PROMPT.format(max_turns=max_turns) + "The game is just initialized."
        next_player = "attacker"

    else:
        query = GAME_RULE_PROMPT.format(max_turns=max_turns) + "\n### Game History:" + history_str
        if history[-1]['role'] == "attacker":
            next_player = "defender"
        else:
            next_player = "attacker"
            
    query += INSTRUCT_PROMPTS[next_player].format(target_word=target_word)
    return query


def randomly_convert_game_history_to_query(history, target_word, max_turns=5):    
    if len(history) == 0:   
        next_player = "attacker"
    else:
        if history[-1]['role'] == "attacker":
            next_player = "defender"
        else:
            next_player = "attacker"

    dialog_prefix = "\n" + random.choice(["\n - ", "\n### ", "\n## ", "\n# ", "\n *** ", "\n **", "\n\n"])
    answer_str, question_str = random.choice([
        (next_player, "defender" if next_player == "attacker" else "attacker"),
        ("Assistant", "Human"),
        ("Answer", "Question"),
        ("Response", "Query"),
        ("A", "Q")
    ])

    player_prefix = {
        "attacker": answer_str if next_player == "attacker" else question_str,
        "defender": answer_str if next_player == "defender" else question_str
    }
    
    history_str = ""
    for i, message in enumerate(history):
        history_str += "{}{}: {}".format(dialog_prefix, player_prefix[message['role']], message['content'])    

    prompt_type = random.choice(['chat', 'chat_inverse', 'alpaca'])
    system_prefix = random.choice(["Rules", "Game Rule", "System"])

    GAME_RULE_PROMPT = random.choice(GAME_RULE_PROMPTS)
    system_prompt = GAME_RULE_PROMPT.format(max_turns=max_turns)
    
    if 'chat' in prompt_type:
        system_prompt += "\n\n" + PLAYER_INSTRUCT_PROMPTS[next_player].format(target_word=target_word)
        
        if len(history) == 0:
            history_str = ""
            system_prompt += "The game is just initialized. "
            
        system_str = f"{dialog_prefix}{system_prefix}: {system_prompt}"
        if "inverse" in prompt_type:
            query = history_str + system_str + dialog_prefix + player_prefix[next_player] + ": "
        else:
            query = system_str + history_str + dialog_prefix + player_prefix[next_player] + ": "
        
    elif prompt_type == "alpaca":
        if random.uniform(0,1) < 0.2:
            system_prompt = system_prefix + ": " + system_prompt
        
        if len(history) == 0:
            query = system_prompt + "The game is just initialized. "
        else:
            query = system_prompt + dialog_prefix + "Game History:" + history_str + '\n\n'

            
        if random.uniform(0,1) < 0.2:
            query += PLAYER_INSTRUCT_PROMPTS[next_player].format(target_word=target_word)[:-1] + ": "
        else:
            query += PLAYER_INSTRUCT_PROMPTS[next_player].format(target_word=target_word) + dialog_prefix + player_prefix[next_player] + ": "
            
    return query


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def read_json_or_jsonl_data(data_path):
    if data_path[-5:] == ".json":
        with open(data_path, 'r') as f:
            data_list = json.load(f)
    else:
        with open(data_path, 'r') as f:
            lines = f.read().strip().split('\n')
            data_list = [json.loads(l) for l in lines]

    print_rank_0(f">>> totally load {len(data_list)} data from {data_path}")
    return data_list


def merge_json_or_jsonl_data(data_path_pattern):
    file_names = glob.glob(data_path_pattern)
    print_rank_0(f"load {len(file_names)} files from {data_path_pattern}.")
    outputs = []
    for file_name in file_names:
        new_data = read_json_or_jsonl_data(file_name)
        if isinstance(new_data, list):
            outputs.extend(new_data)
        elif isinstance(new_data, dict):
            outputs.append(new_data)
    return outputs

    
def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
           
