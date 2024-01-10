"""
Load Wiki-103.
"""
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple, Set

import os
import json
import hydra
from collections import defaultdict
import random
from omegaconf import OmegaConf, DictConfig
import transformers
import resource
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from dpo_utils import (
    get_local_dir,
    TemporarilySeededRandom,
    pad_to_length,
    disable_dropout,
    get_local_run_dir,
)
from dpo_constants import ROOT_DIR, DATA_DIR, GPT2_PAD_IDX

OUTPUT_DIR = os.path.join(DATA_DIR, "toxicity_pairwise_logits")


OmegaConf.register_new_resolver(
    "get_local_run_dir",
    lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs),
)


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]]
) -> Dict[str, torch.LongTensor]:
    """
    Concatenate the positive and negative inputs into a single tensor.

    :params:

    :batch: A batch of data. Must contain the keys 'pos_input_ids' and
        'neg_input_ids', which are tensors of shape (batch, seq).

    :returns:
        A dictionary containing the concatenated inputs under the key
        'concatenated_input_ids'.
    """
    max_length = max(
        batch["pos_input_ids"].shape[1],
        batch["neg_input_ids"].shape[1],
    )
    concatenated_batch = {}
    for k in batch:
        if k.startswith("pos_") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("pos", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(
                batch[k], max_length, pad_value=pad_value
            )
    for k in batch:
        if k.startswith("neg_") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("neg", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            )
    return concatenated_batch


def get_batch_logps(
    logits: torch.FloatTensor,
    input_ids: torch.FloatTensor,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """
    Compute the log probabilities of the given labels under the given logits.

    :params:

    :logits: Logits of the model (unnormalized). (batch, seq, vocab)
    :labels: Labels for which to compute the log probabilities.
        Label tokens with a value of -100 are ignored. (batch, seq)
    :average_log_prob: If True, return the average log probability per
        (non-masked) token. Otherwise, return the sum of the log probabilities
        of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log
        probabilities of the given labels under the given logits.
    """

    # assert logits.shape[:-1] == labels.shape
    # Why index by 1?
    # [batch, seq]
    labels = input_ids[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != GPT2_PAD_IDX

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == GPT2_PAD_IDX] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def iterate_wiki(
    ref_model,
    tokenizer,
    config,
    filename,
    output_filepath,
    device: str = "cuda",
) -> Iterator[Dict]:
    """
    Get an iterator over batches of data.

    :params:

    :split: Which split to use.
    :batch_size: Batch size.
    :valid_size: Validation size.
    """
    batch_size = config.batch_size
    max_prompt_length = config.max_prompt_length
    max_new_tokens = config.max_new_tokens

    with open(filename, "r") as file_p:
        data = file_p.readlines()
    data_size = len(data)

    save_data = []
    for idx in tqdm(range(0, data_size, batch_size)):
        batch = data[idx : idx + batch_size]
        batch = [json.loads(x.strip()) for x in batch]

        prompt_text = [x["prompt_text"] for x in batch]
        # gold_text = [x["gold_text"] for x in batch]
        gold_text = [x["unpert_gen_text"] for x in batch]

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

        batch = {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "pos_text": gold_text,
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_input_ids != tokenizer.pad_token_id,
            "pos_labels": pos_labels,
            "neg_text": pplm_text,
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_input_ids != tokenizer.pad_token_id,
            "neg_labels": neg_labels,
        }
        batch.update(concatenated_inputs(batch))

        with torch.no_grad():
            all_logits = ref_model(
                batch["concatenated_input_ids"],
                attention_mask=batch["concatenated_attention_mask"],
            ).logits.to(torch.float32)
            all_logps = get_batch_logps(
                all_logits,
                batch["concatenated_input_ids"],
                average_log_prob=False,
            )

        num_pos_samples = batch["pos_input_ids"].shape[0]
        batch["pos_logps"] = all_logps[:num_pos_samples]
        batch["neg_logps"] = all_logps[num_pos_samples:]

        for name, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[name] = batch[name].cpu().numpy().tolist()

        save_data.append(json.dumps(batch))

    with open(output_filepath, "w") as file_p:
        for _out in save_data:
            file_p.write(_out)
            file_p.write("\n")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """
    Main entry point for training.
    Validates config, creates/initializes model(s),
    and kicks off worker process(es).
    """
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print("WARNING: eval_every must be divisible by batch_size")
        print(
            "Setting eval_every to",
            config.eval_every - config.eval_every % config.batch_size,
        )
        config.eval_every = (
            config.eval_every - config.eval_every % config.batch_size
        )

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)

    os.environ["XDG_CACHE_HOME"] = get_local_dir(config.local_dirs)

    model_kwargs = {"device_map": "balanced"}
    print("building reference model")
    reference_model_dtype = getattr(torch, config.model.reference_dtype)
    reference_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=get_local_dir(config.local_dirs),
        low_cpu_mem_usage=True,
        torch_dtype=reference_model_dtype,
        **model_kwargs,
    )
    disable_dropout(reference_model)

    tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        "gpt2-medium", cache_dir=get_local_dir(config.local_dirs)
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_dir = os.path.join(DATA_DIR, "toxicity_pairwise/")
    filenames = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".jsonl")
    ]

    for filepath in tqdm(filenames):
        _filename = filepath.split("/")[-1]
        if not _filename.endswith("split_1.jsonl"):
            continue

        output_filepath = os.path.join(OUTPUT_DIR, _filename)

        iterate_wiki(
            reference_model, tokenizer, config, filepath, output_filepath
        )


if __name__ == "__main__":
    main()
