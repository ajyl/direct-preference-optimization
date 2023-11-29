"""
Load Repetitions pairwise data.
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


_DATA_DIR = os.path.join(DATA_DIR, "repetitions/_pairwise")


def load_wiki103(data_type=None):
    """
    Load Wiki103.
    """
    wiki_dir = os.path.join(_DATA_DIR, "wikitext-103")
    if data_type is None:
        data_type = ["train", "valid", "test"]
    prompts = []
    for _type in data_type:
        with open(
            os.path.join(wiki_dir, f"wiki.{_type}.tokens"), "r"
        ) as file_p:
            data = file_p.readlines()

        filtered = []
        for line in data:
            if "=" in line:
                continue

            sentences = line.split(".")
            for sent in sentences:
                if "<unk>" in sent:
                    continue
                if len(sent) < 20:
                    continue

                sent = sent.strip()
                if not sent.endswith("."):
                    sent += "."

                filtered.append(sent)
        prompts.append(filtered)

    return prompts


def get_repetitions_batch_iterator(
    model: nn.Module,
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    n_epochs: int = 1,
    max_prompt_length: int = 128,
    max_new_tokens: int = 64,
    valid_size: int = 128,
    device: str = "cuda",
) -> Iterator[Dict]:
    """
    Get an iterator over batches of data.

    :params:

    :split: Which split to use.
    :batch_size: Batch size.
    :n_epochs: Number of epochs to run for.
    :valid_size: Validation size.
    """
    assert split in ["train", "valid"]
    data_dir = os.path.join(
        DATA_DIR, "repetitions/wiki103_splices_w_next_sents"
    )

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
    for _ in range(n_epochs):
        for filename in tqdm(filenames):
            with open(filename, "r") as file_p:
                file_data = file_p.readlines()

            random.shuffle(file_data)
            data_size = len(file_data)
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
                    "gold_input_ids": gold_tokenized["input_ids"]
                }
