"""
Evaluator for DPO.
"""
from typing import Optional, Dict, List, Union, Tuple

import random
import os
from collections import defaultdict
import time
import json
import contextlib

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import KLDivLoss
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import transformers
from omegaconf import DictConfig

from dpo_toxicity.paradetox_dataset import get_paradetox_batch_iterator
from dpo_toxicity.wiki103_dataset import get_wiki103_batch_iterator
from dpo_toxicity.trainers import (
    get_kl_div,
    get_batch_logps,
    generate,
    get_prec_recall_f1,
    concatenated_inputs,
)
from dpo_utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
from dpo_constants import GPT2_PAD_IDX

torch.backends.cuda.matmul.allow_tf32 = True


class BasicEvaluator(object):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        A trainer for a language model, supporting either SFT or DPO training.

        If multiple GPUs are present, naively splits the model across them, effectively
        offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.example_counter = 0

        tokenizer_name_or_path = (
            config.model.tokenizer_name_or_path or config.model.name_or_path
        )
        rank0_print(f"Loading tokenizer {tokenizer_name_or_path}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs)
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.policy = policy
        self.reference_model = reference_model
        self.kl_criterion = KLDivLoss(reduction="none", log_target=True)

        self.train_iterator = get_paradetox_batch_iterator(
            self.tokenizer,
            self.config,
            split="train",
        )
        self.eval_iterator = get_paradetox_batch_iterator(
            self.tokenizer,
            self.config,
            split="valid",
        )

        self.eval_batches = list(self.eval_iterator)
        rank0_print(
            f"Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}"
        )

        self.wiki103_batches = None
        if config.include_wiki103:
            self.wiki103_iterator = get_wiki103_batch_iterator(
                self.tokenizer,
                self.config,
                split="valid",
            )
            self.wiki103_batches = list(self.wiki103_iterator)

    def get_batch_samples(
        self, batch: Dict[str, torch.LongTensor]
    ) -> Tuple[str, str]:
        """
        Generate samples from the policy (and reference model, if doing DPO training)
        for the given batch of inputs
        ."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(
                self.policy, writeback=False, recurse=False
            )
            if "FSDP" in self.config.evaluator
            else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.policy.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if self.config.loss.name == "dpo":
            ctx = lambda: (
                FSDP.summon_full_params(
                    self.reference_model, writeback=False, recurse=False
                )
                if "FSDP" in self.config.evaluator
                else contextlib.nullcontext()
            )
            with ctx():
                reference_output = self.reference_model.generate(
                    batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.config.max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        policy_output = pad_to_length(
            policy_output, self.config.max_length, self.tokenizer.pad_token_id
        )
        policy_output = all_gather_if_needed(
            policy_output, self.rank, self.world_size
        )
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )

        reference_output_decoded = []
        if self.config.loss.name == "dpo":
            reference_output = pad_to_length(
                reference_output,
                self.config.max_length,
                self.tokenizer.pad_token_id,
            )
            reference_output = all_gather_if_needed(
                reference_output, self.rank, self.world_size
            )
            reference_output_decoded = self.tokenizer.batch_decode(
                reference_output, skip_special_tokens=True
            )

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Run the given model on the given batch of inputs,
        concatenating the positive and negative inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.

        :returns:
        :pos_logps: (batch)
        :neg_logps: (batch)
        :pos_logits: (batch, seq, vocab)
        :neg_logits: (batch, seq, vocab)
        """
        concatenated_batch = concatenated_inputs(batch)

        # [batch (*2), seq (prompt + response), vocab]
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        all_logps = get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_input_ids"],
            average_log_prob=False,
        )

        num_pos_samples = batch["pos_input_ids"].shape[0]
        pos_logps = all_logps[:num_pos_samples]
        neg_logps = all_logps[num_pos_samples:]
        pos_logits = all_logits[:num_pos_samples]
        neg_logits = all_logits[num_pos_samples:]
        return pos_logps, neg_logps, pos_logits, neg_logits

    def get_batch_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        """
        Compute the SFT or DPO loss and other metrics for the given batch of inputs.
        """

        metrics = {}
        train_test = "valid"

        (
            policy_pos_logps,
            policy_neg_logps,
            policy_pos_logits,
            policy_neg_logits,
        ) = self.concatenated_forward(self.policy, batch)
        with torch.no_grad():
            (
                ref_pos_logps,
                ref_neg_logps,
                ref_pos_logits,
                ref_neg_logits,
            ) = self.concatenated_forward(self.reference_model, batch)

        pos_kl_div, neg_kl_div = get_kl_div(
            self.kl_criterion,
            policy_pos_logits,
            policy_neg_logits,
            ref_pos_logits,
            ref_neg_logits,
        )

        metrics[f"kl_div_{train_test}/positive"] = (
            pos_kl_div.detach().cpu().numpy().tolist()
        )

        metrics[f"kl_div_{train_test}/negative"] = (
            neg_kl_div.detach().cpu().numpy().tolist()
        )

        return metrics

    def get_wiki103_metrics(self, batch):
        """
        Get wiki103 specific metrics.
        """
        metrics = {}
        for model in [
            ("policy", self.policy),
            ("reference", self.reference_model),
        ]:
            generations = generate(
                model[1],
                batch,
                self.config.max_new_tokens,
                self.tokenizer.pad_token_id,
                fsdp="FSDP" in self.config.evaluator,
            )
            _batch = batch.copy()
            _batch.update(generations)
            local_batch = slice_and_move_batch_for_device(
                _batch, self.rank, self.world_size, self.rank
            )

            policy_precs = []
            policy_recalls = []
            policy_f1s = []
            for batch_idx in range(local_batch["gold_input_ids"].shape[0]):
                _gold = local_batch["gold_input_ids"][batch_idx].to(self.rank)
                _gold = _gold[_gold != GPT2_PAD_IDX].unique()

                _generated = local_batch["policy_input_ids"][batch_idx]
                _generated = _generated[_generated != GPT2_PAD_IDX].unique()
                neg_prec, neg_recall, neg_f1 = get_prec_recall_f1(
                    _generated, _gold
                )
                policy_precs.append(neg_prec)
                policy_recalls.append(neg_recall)
                policy_f1s.append(neg_f1)
            metrics[
                f"{model[0]}_precision_wiki103_valid/negative"
            ] = policy_precs
            metrics[
                f"{model[0]}_recall_wiki103_valid/negative"
            ] = policy_recalls
            metrics[f"{model[0]}_f1_wiki103_valid/negative"] = policy_f1s

        return metrics

    def eval(self):
        """
        Run evaluation.
        """
        rank0_print(
            f"Running evaluation after {self.example_counter} train examples"
        )
        self.policy.eval()

        standard_eval = self._eval()
        if self.config.include_wiki103:
            self._wiki103_eval()
        return standard_eval

    def _eval(self):
        """
        Run evaluation.
        """
        all_eval_metrics = defaultdict(list)
        if self.config.sample_during_eval:
            all_policy_samples, all_reference_samples = [], []

        for eval_batch in (
            tqdm(self.eval_batches, desc="Computing eval metrics")
            if self.rank == 0
            else self.eval_batches
        ):

            local_eval_batch = slice_and_move_batch_for_device(
                eval_batch, self.rank, self.world_size, self.rank
            )
            with torch.no_grad():
                eval_metrics = self.get_batch_metrics(local_eval_batch)

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        if (
            self.config.sample_during_eval
            and self.example_counter % self.config.sample_every == 0
        ):
            if self.config.n_eval_model_samples < self.config.eval_batch_size:
                rank0_print(
                    f"Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < \
                    eval_batch_size ({self.config.eval_batch_size}). \
                    Sampling from the first complete eval batch of prompts."
                )
                sample_batches = self.eval_batches[:1]
            else:
                n_sample_batches = (
                    self.config.n_eval_model_samples
                    // self.config.eval_batch_size
                )
                sample_batches = self.eval_batches[:n_sample_batches]

            for eval_batch in (
                tqdm(sample_batches, desc="Generating samples...")
                if self.rank == 0
                else sample_batches
            ):
                local_eval_batch = slice_and_move_batch_for_device(
                    eval_batch, self.rank, self.world_size, self.rank
                )
                (
                    policy_samples,
                    reference_samples,
                ) = self.get_batch_samples(local_eval_batch)

                all_policy_samples.extend(policy_samples)
                all_reference_samples.extend(reference_samples)

            rank0_print("Policy samples:")
            rank0_print(json.dumps(all_policy_samples[:10], indent=2))

        mean_eval_metrics = {
            k: sum(v) / len(v) for k, v in all_eval_metrics.items()
        }
        rank0_print(
            f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
        )

    def _wiki103_eval(self):
        """
        Gather some metrics pertaining to wiki103.
        """
        all_eval_metrics = defaultdict(list)
        if self.config.sample_during_eval:
            all_policy_samples, all_reference_samples = [], []

        for eval_batch in (
            tqdm(self.wiki103_batches, desc="Computing wiki103 metrics")
            if self.rank == 0
            else self.wiki103_batches
        ):

            with torch.no_grad():
                wiki103_metrics = self.get_wiki103_metrics(
                    eval_batch,
                )

            for k, v in wiki103_metrics.items():
                all_eval_metrics[k].extend(v)

        if self.config.sample_during_eval_wiki103:
            if self.config.n_eval_model_samples < self.config.eval_batch_size:
                rank0_print(
                    f"Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < \
                    eval_batch_size ({self.config.eval_batch_size}). \
                    Sampling from the first complete eval batch of prompts."
                )
                sample_batches = self.eval_batches[:1]
            else:
                n_sample_batches = (
                    self.config.n_eval_model_samples
                    // self.config.eval_batch_size
                )
                sample_batches = self.eval_batches[:n_sample_batches]

            for eval_batch in (
                tqdm(sample_batches, desc="Generating samples...")
                if self.rank == 0
                else sample_batches
            ):
                local_eval_batch = slice_and_move_batch_for_device(
                    eval_batch, self.rank, self.world_size, self.rank
                )
                (
                    policy_samples,
                    reference_samples,
                ) = self.get_batch_samples(local_eval_batch)

                all_policy_samples.extend(policy_samples)
                all_reference_samples.extend(reference_samples)

            rank0_print("Policy samples:")
            rank0_print(json.dumps(all_policy_samples[:10], indent=2))

        mean_eval_metrics = {
            k: sum(v) / len(v) for k, v in all_eval_metrics.items()
        }
        rank0_print(
            f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
        )
