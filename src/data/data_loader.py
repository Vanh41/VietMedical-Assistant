import json
import os 
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def load_json_or_jsonl(path):
    """Load .json or .jsonl dataset"""
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return data
    
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    raise ValueError("Dataset must be .json or .jsonl")

def prepare_dataset(data_path, max_length=2048, train_split=0.9):
    """Load for SFT

    Args:
        data_path (_type_): _description_
        tokenizer (_type_): _description_
        text_field (str, optional): _description_. Defaults to "text".
        max_length (int, optional): _description_. Defaults to 2048.
        train_split (float, optional): _description_. Defaults to 0.9.
    """
    
    raw_data = load_json_or_jsonl(data_path)
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.train_test_split(test_size=1 - train_split, shuffle=True, seed=42)
    return dataset


if __name__ == "__main__":
    dataset = prepare_dataset(
        data_path="data/raw/viet_med_qa.jsonl",
    )
    print(dataset)

