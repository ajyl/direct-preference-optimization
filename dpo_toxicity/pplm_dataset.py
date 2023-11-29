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


def get_pplm_batch_iterator(
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
    data_dir = os.path.join(DATA_DIR, "toxicity/wiki103_toxicity_pairwise")
    batch_size = config.batch_size
    if split == "valid":
        batch_size = config.eval_batch_size
    max_prompt_length = config.max_prompt_length
    max_new_tokens = config.max_new_tokens
    valid_size = config.valid_size

    filenames = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".jsonl")
    ]

    data = []
    for filename in tqdm(filenames):
        with open(filename, "r") as file_p:
            file_data = file_p.readlines()

        data.extend(file_data)

    random.shuffle(file_data)
    if split == "train":
        data = data[:-valid_size]
    else:
        data = data[-valid_size:]
    data_size = len(data)

    start = 0
    if split == "train":
        start = 4092
    for idx in range(start, data_size, batch_size):
        batch = data[idx : idx + batch_size]
        batch = [json.loads(x.strip()) for x in batch]

        prompt_text = [x["prompt_text"] for x in batch]
        gold_text = [x["gold_text"] for x in batch]

        prompt_tokenized = tokenizer(
            prompt_text,
            max_length=max_prompt_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        prompt_input_ids = prompt_tokenized["input_ids"]
        prompt_attention_mask = prompt_tokenized["attention_mask"]

        tokenizer.padding_side = "right"
        gold_tokenized = tokenizer(
            gold_text,
            max_length=max_new_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        pos_input_id = gold_tokenized["input_ids"].long()

        pplm_text = [x["pert_gen_text"] for x in batch]
        pplm_tokenized = tokenizer(
            pplm_text,
            max_length=max_new_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        tokenizer.padding_side = "left"

        pos_input_ids = torch.concat(
            [prompt_input_ids, gold_tokenized["input_ids"]], dim=1
        )
        neg_input_ids = torch.concat(
            [prompt_input_ids, pplm_tokenized["input_ids"]], dim=1
        )

        prompt_shape = prompt_input_ids.shape[1]
        pos_labels = pos_input_ids.detach().clone()
        pos_labels[:, :prompt_shape] = -100
        neg_labels = neg_input_ids.detach().clone()
        neg_labels[:, :prompt_shape] = -100

        yield {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "gold_text": gold_text,
            "gold_input_ids": pos_input_id,
            "pos_text": gold_text,
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_input_ids != tokenizer.pad_token_id,
            "pos_labels": pos_labels,
            "neg_text": pplm_text,
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_input_ids != tokenizer.pad_token_id,
            "neg_labels": neg_labels,
        }