"""
Load Repetitions pairwise data.
"""
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple

import os
from collections import defaultdict
import random
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from dpo_utils import get_local_dir, TemporarilySeededRandom
from dpo_constants import DATA_DIR, GPT2_PAD_IDX


_DATA_DIR = os.path.join(DATA_DIR, "repetitions/_pairwise")


def load_data(valid_size):
    """
    Load data.
    """
    # data = torch.load(os.path.join(_DATA_DIR, "pairwise_data.pt"))
    data = torch.load(os.path.join(_DATA_DIR, "pairwise_98.pt"))
    # data_size = data.shape[0]
    # idxs = torch.randperm(data_size)

    train = data[:-valid_size, ...]
    valid = data[-valid_size:, ...]
    return train, valid


def load_wiki103(data_type = None):
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
    split: str = "train",
    batch_size: int = 1,
    n_epochs: int = 1,
    max_prompt_length: int = 128,
    valid_size: int = 128,
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
    response_dir = os.path.join(DATA_DIR, "repetitions/wiki103_outputs")

    task_number = 98
    prompt_file = os.path.join(response_dir, f"_prompts_{task_number}.pt")
    greedy_file = os.path.join(
        response_dir, f"_greedy_generations_{task_number}.pt"
    )
    ngram_blocked_file = os.path.join(
        response_dir, f"_ngram_blocked_generations_{task_number}.pt"
    )

    prompts = torch.load(prompt_file).to("cpu")
    greedy = torch.load(greedy_file).to("cpu")
    ngram_blocked = torch.load(ngram_blocked_file).to("cpu")

    assert prompts.shape == greedy.shape
    assert greedy.shape == ngram_blocked.shape
    data_size = prompts.shape[0]
    seq_len = prompts.shape[1]

    breakpoint()
    for _ in range(n_epochs):
        for idx in range(0, data_size, batch_size):
            idxs = torch.arange(idx, idx + batch_size)
            batch = data[idxs]

            prompt_ids = batch[:, 0, :]
            _pos_ids = batch[:, 1, :]
            _neg_ids = batch[:, 2, :]

            pos_ids = torch.concat([prompt_ids, _pos_ids], dim=1)
            neg_ids = torch.concat([prompt_ids, _neg_ids], dim=1)

            pos_labels = torch.concat([
                torch.ones_like(prompt_ids) * -100,
                _pos_ids
            ], dim=1)
            neg_labels = torch.concat([
                torch.ones_like(prompt_ids) * -100,
                _neg_ids
            ], dim=1)
            breakpoint()

            yield {
                "prompt_input_ids": prompt_ids,
                "prompt_attention_mask": prompt_ids != GPT2_PAD_IDX,

                "pos_input_ids": pos_ids,
                "pos_attention_mask": pos_ids != GPT2_PAD_IDX,
                "pos_labels": pos_labels,

                "neg_input_ids": neg_ids,
                "neg_attention_mask": neg_ids != GPT2_PAD_IDX,
                "neg_labels": neg_labels,
            }
