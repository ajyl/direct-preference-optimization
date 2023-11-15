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


def get_repetitions_batch_iterator(
    split: str = "train",
    batch_size: int = 1,
    n_epochs: int = 1,
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
    data = load_data(valid_size)
    if split == "train":
        data = data[0]
    elif split == "valid":
        data = data[1]

    data_size = data.shape[0]
    for _ in range(n_epochs):
        for idx in range(0, data_size, batch_size):
            idxs = torch.arange(idx, idx + batch_size)
            batch = data[idxs]

            prompt_ids = batch[:, 0, :]
            pos_ids = batch[:, 1, :]
            neg_ids = batch[:, 2, :]
            breakpoint()

            yield {
                "prompt_input_ids": prompt_ids,
                "prompt_attention_mask": prompt_ids != GPT2_PAD_IDX,
                "pos_input_ids": pos_ids,
                "pos_attention_mask": pos_ids != GPT2_PAD_IDX,
                "neg_input_ids": neg_ids,
                "neg_attention_mask": neg_ids != GPT2_PAD_IDX,
            }
