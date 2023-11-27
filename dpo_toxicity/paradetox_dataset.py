"""
ParaDetox
"""
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple

import os
import csv
import random

random.seed(42)

_DATA_DIR = os.path.dirname(os.path.realpath(__file__))


def get_paradetox_batch_iterator(
    tokenizer,
    config,
    split: str,
    device: str = "cuda",
) -> Iterator[Dict]:

    assert split in ["train", "valid"]
    batch_size = config.batch_size
    max_prompt_length = config.max_prompt_length
    n_epochs = config.n_epochs
    valid_size = config.valid_size
    data_filepath = os.path.join(_DATA_DIR, config.data.filename)

    with open(data_filepath, "r") as file_p:
        reader = csv.reader(file_p, delimiter="\t")
        data = list(reader)

    random.shuffle(data)
    if split == "train":
        data = data[:-valid_size]
    else:
        data = data[-valid_size:]

    data_size = len(data)
    for _ in range(n_epochs):

        for idx in range(1, data_size, batch_size):

            batch = data[idx : idx + batch_size]

            toxic_text = [x[0] for x in batch]
            nontoxic_text = [x[1] for x in batch]

            toxic_tokenized = tokenizer(
                toxic_text,
                max_length=max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            nontoxic_tokenized = tokenizer(
                nontoxic_text,
                max_length=max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            yield {
                "neg_input_ids": toxic_tokenized["input_ids"],
                "neg_attention_mask": toxic_tokenized["attention_mask"],
                "pos_input_ids": nontoxic_tokenized["input_ids"],
                "pos_attention_mask": nontoxic_tokenized["attention_mask"],
            }
