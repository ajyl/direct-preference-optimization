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
from dpo_toxicity.toxic_utils import TOXICITY_METRICS


_DATA_DIR = os.path.join(
    DATA_DIR, "realtoxicityprompts-data/generations/prompted/"
)

random.seed(42)


def get_toxicity_batch_iterator(
    model: nn.Module,
    tokenizer,
    config,
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
    data_filepath = os.path.join(_DATA_DIR, config.data.filepath)

    with open(data_filepath, "r") as file_p:
        data = file_p.readlines()
    random.shuffle(data)
    if split == "train":
        data = data[:-valid_size]
    else:
        data = data[-valid_size:]

    for _ in range(n_epochs):
        data_size = len(data)
        for idx in range(0, data_size, batch_size):
            batch = data[idx : idx + batch_size]

            # {
            #   "filename": str,
            #   "begin": int,
            #   "end": int,
            #   "prompt": {
            #       "text": str,
            #       metrics
            #   },
            #   "continuation"; {
            #       "text"; str,
            #       metrics
            #   },
            #   "generations": [
            #       {
            #           "text": str
            #           metrics
            #       }, ...
            #   ]
            # }
            batch = [json.loads(x.strip()) for x in batch]

            prompts = [x["prompt"] for x in batch]
            prompt_texts = [x["text"] for x in prompts]
            # prompt_scores = [
            #    {metric: x[metric] for metric in TOXICITY_METRICS}
            #    for x in prompts
            # ]

            continuations = [x["continuation"] for x in batch]
            continuation_texts = [x["text"] for x in continuations]
            continuation_scores = [
                {metric: x[metric] for metric in TOXICITY_METRICS}
                for x in continuations
            ]

            # List of lists.
            generations = [x["generations"] for x in batch]
            generation_texts = [
                [x["text"] for x in _generations]
                for _generations in generations
            ]
            generation_scores = [
                [
                    {
                        metric: x[metric]
                        for metric in TOXICITY_METRICS
                        if x[metric] is not None
                    }
                    for x in _generations
                ]
                for _generations in generations
            ]

            generation_max_scores = []
            for batch_idx, _scores in enumerate(generation_scores):
                max_scores = []
                for idx, metrics in enumerate(_scores):
                    values = [v for v in metrics.values() if v is not None]
                    max_score = np.nan
                    if len(values) > 0:
                        max_score = max(values)
                    max_scores.append(max_score)

                max_continuation_score = np.nan
                _continuation_scores = list(continuation_scores[batch_idx].values())
                if len(_continuation_scores) > 0:
                    max_continuation_score = max(_continuation_scores)
                max_scores.append(max_continuation_score)

                generation_max_scores.append(np.array(max_scores))

            max_idxs = [np.nanargmax(x) for x in generation_max_scores]
            min_idxs = [np.nanargmin(x) for x in generation_max_scores]
            breakpoint()
            continuation_max_scores = [
                max([_x for _x in x.values()]) for x in continuation_scores
            ]
            scores = [
                np.array(scores + [continuation_max_scores[_idx]])
                for _idx, scores in enumerate(generation_max_scores)
            ]
            breakpoint()

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
