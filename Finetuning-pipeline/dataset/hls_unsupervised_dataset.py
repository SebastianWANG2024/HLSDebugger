from datasets import load_dataset
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, cast
import transformers

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
# Load the dataset and format it for training.
# data = load_dataset("Abirate/english_quotes", split="train")

#Load the customised dataset
data = load_dataset(
    'json',
    data_files="/home/jwangjw/HLS_Codes/Unlabelled_Dataset_v0/unlabeld_code_dataset_split_30.jsonl",
    split="train",
    cache_dir="/home/jwangjw/HLS_Codes/Unlabelled_Dataset_v0/cache"
    )
# print(data['train'][:5])
print(data.column_names)
print(len(data['code']))

def preprocess(
        dataset: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    tokenized_codes = _tokenize_fn(dataset['code'], tokenizer)
    return dict(
        input_ids = tokenized_codes['input_ids'],
        labels = tokenized_codes['labels'],
    )

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

tokenizer = transformers.AutoTokenizer.from_pretrained(
        "microsoft/codebert-base",
        cache_dir="./test_cache",
        model_max_length=2048,
        padding_side="left",
        use_fast=False,
    )

    
dataset = load_dataset(
    'json',
    data_files="/home/jwangjw/HLS_Codes/Unlabelled_Dataset_v0/unlabeld_code_dataset_split_30.jsonl",
    split="train",
    cache_dir="/home/jwangjw/HLS_Codes/Unlabelled_Dataset_v0/cache"
    )

tokenized_datasets = dataset.map(lambda examples: preprocess(examples, tokenizer), batched=True)

print(tokenized_datasets['input_ids'][:5])