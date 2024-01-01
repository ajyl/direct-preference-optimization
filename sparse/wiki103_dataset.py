"""
Load Wiki-103.
"""
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple

import os
import json
from collections import defaultdict
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from dpo_utils import get_local_dir, TemporarilySeededRandom
from dpo_constants import DATA_DIR, GPT2_PAD_IDX


def get_wiki103_batch_iterator(
    tokenizer,
    config,
    split: str = "train",
    device: str = "cuda",
) -> Iterator[Dict]:
    """
    Get an iterator over batches of data.

    :params:

    :split: Which split to use.
    :batch_size: Batch size.
    :valid_size: Validation size.
    """
    assert split in ["train", "valid"]
    data_dir = os.path.join(
        DATA_DIR, "wiki103_splices_w_next_sents"
    )
    batch_size = config.batch_size
    max_prompt_length = config.max_prompt_length
    max_new_tokens = config.max_new_tokens
    valid_size = config.valid_size

    filenames = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".jsonl")
    ]
    if split == "train":
        filenames = filenames[:-1]
    else:
        filenames = [filenames[-1]]

    random.shuffle(filenames)
    for filename in tqdm(filenames):
        with open(filename, "r") as file_p:
            file_data = file_p.readlines()

        random.shuffle(file_data)
        data_size = len(file_data)
        data_size = 50
        if split == "valid":
            data_size = valid_size

        for idx in range(0, data_size, batch_size):
            data = file_data[idx : idx + batch_size]
            data = [json.loads(x.strip()) for x in data]

            prompt = [x[0] for x in data]
            gold = [x[1] for x in data]

            prompt_tokenized = tokenizer(
                prompt,
                max_length=max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            prompt_input_ids = prompt_tokenized["input_ids"]
            prompt_attention_mask = prompt_tokenized["attention_mask"]

            gold_tokenized = tokenizer(
                gold,
                max_length=max_new_tokens,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            yield {
                "prompt_input_ids": prompt_input_ids,
                "prompt_attention_mask": prompt_attention_mask,
                "gold_text": gold,
                "gold_input_ids": gold_tokenized["input_ids"],
            }
