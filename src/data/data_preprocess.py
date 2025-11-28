import json
import os 
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from data_loader import prepare_dataset

def format_with_system_prompt(example):
    example['conversations'] = [
        {"role": "user", "content": example['question']},
        {"role": "assistant", "content": example['answer']}
    ]
    return example



def formatting_prompts_func(examples, tokenizer):
   convos = examples['conversations']
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
   return { "text" : texts, }

