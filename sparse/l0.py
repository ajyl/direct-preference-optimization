"""
Train loop for DPO.
"""
from typing import Optional, Dict, List, Union, Tuple

import random
import os
from collections import defaultdict
import time
import json
import functools
import contextlib
from collections import Counter

import numpy as np
import wandb
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
import einops
from omegaconf import DictConfig

from sparse.pplm_dataset import get_cached_pplm_batch_iterator
from sparse.toxicity_dataset import (
    get_toxic_prompts_batch_iterator,
    get_toxic_token_ids,
)
from sparse.eval import eval_perplexity
from dpo_utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
from sparse.sparse_utils import reshape_group_mask, concrete_stretched
from dpo_constants import GPT2_PAD_IDX

torch.backends.cuda.matmul.allow_tf32 = True


def generate(
    model,
    batch,
    max_new_tokens,
    pad_token_id,
    include_ngram_blocked=False,
    include_ref=False,
    fsdp=False,
    ref_model=None,
):
    """
    Return greedy and n-gram blocked generations.
    """
    prompt_shape = batch["prompt_input_ids"].shape[1]
    with torch.no_grad():
        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(model, writeback=False, recurse=False)
            if fsdp
            else contextlib.nullcontext()
        )
        with ctx():
            greedy_resp = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        greedy_resp_labels = greedy_resp.detach().clone()
        greedy_resp_labels[:, :prompt_shape] = -100
        output = {
            "policy_input_ids": greedy_resp,
            "policy_attention_mask": greedy_resp != GPT2_PAD_IDX,
            "policy_labels": greedy_resp_labels,
        }

    return output


def dpo_loss(
    policy_pos_logps: torch.FloatTensor,
    policy_neg_logps: torch.FloatTensor,
    ref_pos_logps: torch.FloatTensor,
    ref_neg_logps: torch.FloatTensor,
    beta: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Compute the DPO loss for a batch of policy and reference model log probabilities.

    :params:

    :policy_pos_logps: logprobs of positive responses from policy model: (batch_size,)
    :policy_neg_logps: logprobs of negative responses from policy model: (batch_size,)
    :ref_pos_logps: logprobs of positive responses from reference model: (batch_size,)
    :ref_neg_logps: logprobs of negative responses from reference model: (batch_size,)
    :beta: Temperature parameter for the DPO loss, typically something
        in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.

    :returns:

    A tuple of three tensors: (losses, pos_rewards, neg_rewards).
    The losses tensor contains the DPO loss for each example in the batch.
    The pos_rewards and neg_rewards tensors contain the rewards for the
        positive and neg responses, respectively.
    """
    pi_logratios = policy_pos_logps - policy_neg_logps
    ref_logratios = ref_pos_logps - ref_neg_logps

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    pos_rewards = beta * (policy_pos_logps - ref_pos_logps).detach()
    neg_rewards = beta * (policy_neg_logps - ref_neg_logps).detach()

    return losses, pos_rewards, neg_rewards


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


class BasicTrainer(object):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
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
        if self.config.gradient_accumulation_steps != 1:
            breakpoint()

        self.run_dir = run_dir
        self.example_counter = 0
        self.batch_counter = 0
        self.last_log = None
        self.patience = 0
        self.val_metric_value = -1
        if config.validation_direction == "max":
            self.val_direction = 1
            self.best_val_metric = -1

        else:
            self.val_direction = -1
            self.best_val_metric = 1e10

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

        self.model = policy
        self.model.eval()
        self.orig = {}
        for name, param in self.model.named_parameters():
            self.orig[name] = param.detach().cpu()

        self.train_iterator = get_cached_pplm_batch_iterator(
            self.config,
            self.tokenizer.pad_token_id,
            split="train",
        )
        self.eval_iterator = get_cached_pplm_batch_iterator(
            self.config,
            self.tokenizer.pad_token_id,
            split="valid",
        )
        self.eval_batches = list(self.eval_iterator)
        rank0_print(
            f"Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}"
        )

        self.toxic_data_batches = None
        self.toxic_tokens = None
        if config.include_toxic:
            self.toxic_data_iterator = get_toxic_prompts_batch_iterator(
                self.tokenizer, self.config.valid_size
            )
            self.toxic_data_batches = list(self.toxic_data_iterator)
            self.toxic_tokens = get_toxic_token_ids(self.tokenizer)

        self.alpha_optimizer = None
        self.alpha_scheduler = None

        self.sparse_alpha = config.sparse_alpha
        self.indiv_alphas = config.indiv_alphas
        self.group_alphas = config.group_alphas

        self.diff_params = []
        self.diff_params_map = {}
        self.alpha_params = []
        self.alpha_params_map = {}
        self.alpha_groups = []
        self.alpha_groups_map = {}

        self.init_diff()

        if self.indiv_alphas is False and self.group_alphas is False:
            self.sparse_alpha = 0

        if self.indiv_alphas:
            self.init_indiv_mask()

        if self.group_alphas:
            self.init_grouped_mask()

            # mask size should be 346387
            # mask_size = 0
            # for name, mask in self.alpha_groups_map.items():
            #    mask_size += mask.numel()

        self.concrete_lower = config.concrete_lower
        self.concrete_upper = config.concrete_upper
        assert self.concrete_lower != 0
        assert self.concrete_upper != 0
        self.log_ratio = np.log(-self.concrete_lower / self.concrete_upper)

    def init_diff(self):
        """
        Init diff stuff.
        """
        for name, param in self.model.named_parameters():
            diff_param = torch.zeros_like(param.data)
            diff_param.requires_grad = True
            self.diff_params.append(diff_param)
            self.diff_params_map[name] = diff_param

    def init_indiv_mask(self):
        """
        Init indiv sparse mask.
        """
        for name, param in self.model.named_parameters():
            alpha = (
                torch.zeros_like(param.data, dtype=torch.bfloat16)
                + self.config.alpha_init
            )
            alpha.requires_grad = True
            self.alpha_params_map[name] = alpha
            self.alpha_params.append(alpha)

    def init_grouped_mask(self):
        """
        Init group mask.
        """

        def _add_alpha(name, _shape):
            _alpha = torch.zeros((_shape)) + self.config.alpha_init
            _alpha.requires_grad = True
            _alpha.grad = torch.zeros_like(_alpha)

            self.alpha_groups_map[name] = _alpha
            self.alpha_groups.append(_alpha)

        for name, param in self.model.named_parameters():

            # MLP
            if ".mlp." in name:
                if "c_fc.weight" in name:
                    # [d_model, d_mlp]
                    _add_alpha(name, param.shape[1])

                elif "c_fc.bias" in name:
                    # [d_mlp]
                    _add_alpha(name, 1)

                elif "c_proj.weight" in name:
                    # [d_mlp, d_model]
                    _add_alpha(name, param.shape[0])

                elif "c_proj.bias" in name:
                    _add_alpha(name, 1)

                else:
                    raise RuntimeError("Unexpected group mask.")

            elif ".attn." in name:
                if "c_attn.bias" in name:
                    # [d_model * 3]
                    _add_alpha(name, 1)

                elif "c_attn.weight" in name:
                    num_heads = self.model.config.n_head
                    ## [d_model, d_model * 3] -> [d_model, 3 * num_heads * d_attn]
                    w_q, w_k, w_v = torch.tensor_split(param, 3, dim=1)

                    ## [heads, d_model, d_attn]
                    w_q = einops.rearrange(
                        w_q, "m (i h) -> i m h", i=num_heads
                    )
                    w_k = einops.rearrange(
                        w_k, "m (i h) -> i m h", i=num_heads
                    )
                    w_v = einops.rearrange(
                        w_v, "m (i h) -> i m h", i=num_heads
                    )
                    for head_idx in range(num_heads):
                        _add_alpha(f"{name}.q.{head_idx}", w_q.shape[2])
                        _add_alpha(f"{name}.k.{head_idx}", w_q.shape[2])
                        _add_alpha(f"{name}.v.{head_idx}", w_q.shape[2])
                    # _add_alpha(name, 1)

                elif "c_proj.bias" in name:
                    # [d_model]
                    _add_alpha(name, 1)

                elif "c_proj.weight" in name:
                    # [d_model, d_model]
                    _add_alpha(name, param.shape[1])

                else:
                    raise RuntimeError("Unexpected group mask.")

            elif "ln_1" in name or "ln_2" in name or "ln_f" in name:
                # Note: this if-case also handles bias params.
                _add_alpha(name, 1)

            elif "wte" in name:
                # [vocab, d_model]
                _add_alpha(name, param.shape[0])

            elif "wpe" in name:
                # [pos, d_model]
                _add_alpha(name, param.shape[0])

            else:
                raise RuntimeError("Unexpected group mask.")

    def set_policy_weights(self):
        """
        Set policy weights.
        """
        grad_params = {}
        l0_pen = 0
        nonzero_params = 0
        nonzero_z = 0
        nonzero_groups = 0

        for name, param in self.model.named_parameters():

            _z = torch.ones((1))
            _z_grad = torch.ones((1))
            _grouped_z = torch.ones((1))
            _grouped_z_grad = torch.ones((1))
            _reshaped_grouped_z = _grouped_z

            if self.indiv_alphas:
                # Alpha params
                alpha = self.alpha_params_map[name]
                alpha.grad = torch.zeros_like(alpha.data)

                _z, _z_grad = concrete_stretched(
                    alpha, self.concrete_lower, self.concrete_upper
                )
                if l0_pen == 0:
                    l0_pen += torch.sigmoid(alpha - self.log_ratio).sum()
                else:
                    l0_pen += (
                        torch.sigmoid(alpha - self.log_ratio)
                        .sum()
                        .to(l0_pen.device)
                    )

            if self.group_alphas:
                # alpha_group = self.alpha_groups_map[name]
                # alpha_group.grad = torch.zeros_like(alpha_group.data)
                # _grouped_z, _grouped_z_grad = concrete_stretched(
                #    alpha_group, self.concrete_lower, self.concrete_upper
                # )
                # if l0_pen == 0:
                #    l0_pen += torch.sigmoid(alpha_group - self.log_ratio).sum()
                # else:
                #    l0_pen += (
                #        torch.sigmoid(alpha_group - self.log_ratio)
                #        .sum()
                #        .to(l0_pen.device)
                #    )

                if name.endswith("attn.c_attn.weight"):

                    for i in range(self.model.config.n_head):
                        self.alpha_groups_map[
                            f"{name}.q.{i}"
                        ].grad = torch.zeros_like(
                            self.alpha_groups_map[f"{name}.q.{i}"]
                        )
                        self.alpha_groups_map[
                            f"{name}.k.{i}"
                        ].grad = torch.zeros_like(
                            self.alpha_groups_map[f"{name}.k.{i}"]
                        )
                        self.alpha_groups_map[
                            f"{name}.v.{i}"
                        ].grad = torch.zeros_like(
                            self.alpha_groups_map[f"{name}.v.{i}"]
                        )

                    w_q = torch.concat(
                        [
                            self.alpha_groups_map[f"{name}.q.{head_idx}"]
                            for head_idx in range(self.model.config.n_head)
                        ],
                        dim=0,
                    )
                    w_k = torch.concat(
                        [
                            self.alpha_groups_map[f"{name}.k.{head_idx}"]
                            for head_idx in range(self.model.config.n_head)
                        ],
                        dim=0,
                    )
                    w_v = torch.concat(
                        [
                            self.alpha_groups_map[f"{name}.v.{head_idx}"]
                            for head_idx in range(self.model.config.n_head)
                        ],
                        dim=0,
                    )

                    alpha_group = torch.concat([w_q, w_k, w_v], dim=0)
                    alpha_group.grad = torch.zeros_like(alpha_group.data)
                    _grouped_z, _grouped_z_grad = concrete_stretched(
                        alpha_group,
                        self.concrete_lower,
                        self.concrete_upper,
                    )

                    if l0_pen == 0:
                        l0_pen += torch.sigmoid(
                            alpha_group - self.log_ratio
                        ).sum()
                    else:
                        l0_pen += (
                            torch.sigmoid(alpha_group - self.log_ratio)
                            .sum()
                            .to(l0_pen.device)
                        )

                else:
                    alpha_group = self.alpha_groups_map[name]
                    alpha_group.grad = torch.zeros_like(alpha_group.data)
                    _grouped_z, _grouped_z_grad = concrete_stretched(
                        alpha_group, self.concrete_lower, self.concrete_upper
                    )
                    if l0_pen == 0:
                        l0_pen += torch.sigmoid(
                            alpha_group - self.log_ratio
                        ).sum()
                    else:
                        l0_pen += (
                            torch.sigmoid(alpha_group - self.log_ratio)
                            .sum()
                            .to(l0_pen.device)
                        )

            mask = _z
            if self.group_alphas:
                _reshaped_grouped_z, _grouped_z_grad = reshape_group_mask(
                    name,
                    param,
                    _grouped_z,
                    _grouped_z_grad,
                    self.model.config,
                )
                mask = _reshaped_grouped_z

            if self.indiv_alphas and self.group_alphas:
                mask = _z * _reshaped_grouped_z.to(_z.device)

            # Diff params
            diff = self.diff_params_map[name]
            diff.grad = torch.zeros_like(param.data)
            mask = mask.to(diff.device)

            ref_weight = self.orig[name]

            if self.indiv_alphas or self.group_alphas:
                param.data.copy_(ref_weight.to(diff.device) + mask * diff.data)
            else:
                param.data.copy_(ref_weight.to(diff.device) + diff.data)

            nonzero_z += (_z > 0).detach().sum().float().item()
            nonzero_params += (mask > 0).detach().sum().float().item()
            if self.group_alphas:
                nonzero_groups += (
                    (_grouped_z > 0).detach().sum().float().item()
                )

            grad_params[name] = {
                "z_grad": _z_grad,
                "group_z_grad": _grouped_z_grad,
                "dense_params": diff,
                "z": _z,
                "group_z": _reshaped_grouped_z,
            }

        results = {
            "nonzero_params": nonzero_params,
            "nonzero_z": nonzero_z,
            "nonzero_groups": nonzero_groups,
            "l0": l0_pen,
            "grad_params": grad_params,
        }
        return results

    def set_grads(self, grad_params):
        """
        Set gradients.
        """
        for name, param in self.model.named_parameters():
            if param.grad is None:
                breakpoint()

            diff_param = grad_params[name]["dense_params"]
            z_mask = grad_params[name]["z"]
            group_z_mask = grad_params[name]["group_z"]
            group_z_grad = grad_params[name]["group_z_grad"]
            z_grad = grad_params[name]["z_grad"]

            mask = z_mask
            if self.group_alphas:
                mask = group_z_mask.to(param.device)
            if self.indiv_alphas and self.group_alphas:
                mask = z_mask * group_z_mask.to(z_mask.device)

            self.diff_params_map[name].grad.copy_(
                param.grad.data * mask.to(param.device)
            )
            if self.indiv_alphas:
                self.alpha_params_map[name].grad.copy_(
                    param.grad.data * diff_param * z_mask * z_grad
                )
            if self.group_alphas:
                # self.alpha_groups_map[name].grad.copy_(
                #    (
                #        param.grad.data
                #        * diff_param.data
                #        * group_z_mask.to(param.device)
                #        * group_z_grad.to(param.device)
                #    ).sum()
                # )

                if "c_attn.weight" in name:
                    num_heads = self.model.config.n_head
                    q_param, k_param, v_param = torch.tensor_split(
                        param.grad.data, 3, dim=1
                    )
                    q_diff, k_diff, v_diff = torch.tensor_split(
                        diff_param, 3, dim=1
                    )
                    q_mask, k_mask, v_mask = torch.tensor_split(
                        group_z_mask, 3, dim=1
                    )
                    q_z_grad, k_z_grad, v_z_grad = torch.tensor_split(
                        group_z_grad, 3, dim=1
                    )

                    q_param = einops.rearrange(
                        q_param, "m (i h) -> i m h", i=num_heads
                    )
                    k_param = einops.rearrange(
                        k_param, "m (i h) -> i m h", i=num_heads
                    )
                    v_param = einops.rearrange(
                        v_param, "m (i h) -> i m h", i=num_heads
                    )

                    q_diff = einops.rearrange(
                        q_diff, "m (i h) -> i m h", i=num_heads
                    )
                    k_diff = einops.rearrange(
                        k_diff, "m (i h) -> i m h", i=num_heads
                    )
                    v_diff = einops.rearrange(
                        v_diff, "m (i h) -> i m h", i=num_heads
                    )

                    q_mask = einops.rearrange(
                        q_mask, "m (i h) -> i m h", i=num_heads
                    )
                    k_mask = einops.rearrange(
                        k_mask, "m (i h) -> i m h", i=num_heads
                    )
                    v_mask = einops.rearrange(
                        v_mask, "m (i h) -> i m h", i=num_heads
                    )

                    q_z_grad = einops.rearrange(
                        q_z_grad, "m (i h) -> i m h", i=num_heads
                    )
                    k_z_grad = einops.rearrange(
                        k_z_grad, "m (i h) -> i m h", i=num_heads
                    )
                    v_z_grad = einops.rearrange(
                        v_z_grad, "m (i h) -> i m h", i=num_heads
                    )

                    for i in range(num_heads):
                        self.alpha_groups_map[f"{name}.q.{i}"].grad.copy_(
                            (
                                q_param[i]
                                * q_diff[i]
                                * q_mask[i].to(q_param.device)
                                * q_z_grad[i].to(q_param.device)
                            ).sum()
                        )

                        self.alpha_groups_map[f"{name}.k.{i}"].grad.copy_(
                            (
                                k_param[i]
                                * k_diff[i]
                                * k_mask[i].to(k_param.device)
                                * k_z_grad[i].to(k_param.device)
                            ).sum()
                        )

                        self.alpha_groups_map[f"{name}.v.{i}"].grad.copy_(
                            (
                                v_param[i]
                                * v_diff[i]
                                * v_mask[i].to(v_param.device)
                                * v_z_grad[i].to(v_param.device)
                            ).sum()
                        )

                else:
                    self.alpha_groups_map[name].grad.copy_(
                        (
                            param.grad.data
                            * diff_param
                            * group_z_mask.to(param.device)
                            * group_z_grad.to(param.device)
                        ).sum()
                    )

    def get_batch_samples(
        self, batch: Dict[str, torch.LongTensor]
    ) -> Tuple[str, str]:
        """
        Generate samples from the policy (and reference model, if doing DPO training)
        for the given batch of inputs
        """

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(
                self.policy, writeback=False, recurse=False
            )
            if "FSDP" in self.config.trainer
            else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.model.generate(
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

        return policy_output_decoded

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
        loss_config: DictConfig,
        train=True,
    ):
        """
        Compute the SFT or DPO loss and other metrics for the given batch of inputs.
        """

        metrics = {}
        train_test = "train" if train else "valid"
        grad_params = self.set_policy_weights()

        (
            policy_pos_logps,
            policy_neg_logps,
            policy_pos_logits,
            policy_neg_logits,
        ) = self.concatenated_forward(self.model, batch)

        ref_pos_logps = batch["ref_pos_logps"]
        ref_neg_logps = batch["ref_neg_logps"]

        losses, pos_rewards, neg_rewards = dpo_loss(
            policy_pos_logps,
            policy_neg_logps,
            ref_pos_logps,
            ref_neg_logps,
            beta=loss_config.beta,
        )

        metrics[f"dpo_loss_{train_test}"] = (
            losses.detach().cpu().numpy().tolist()
        )

        if self.indiv_alphas:
            losses = losses + self.sparse_alpha * grad_params["l0"]
            metrics[f"summed_loss_{train_test}"] = (
                losses.detach().cpu().numpy().tolist()
            )

        reward_accuracies = (pos_rewards > neg_rewards).float()

        pos_rewards = all_gather_if_needed(
            pos_rewards, self.rank, self.world_size
        )
        neg_rewards = all_gather_if_needed(
            neg_rewards, self.rank, self.world_size
        )
        reward_accuracies = all_gather_if_needed(
            reward_accuracies, self.rank, self.world_size
        )

        metrics[f"rewards_{train_test}/positive"] = (
            pos_rewards.cpu().numpy().tolist()
        )
        metrics[f"rewards_{train_test}/negative"] = (
            neg_rewards.cpu().numpy().tolist()
        )
        metrics[f"rewards_{train_test}/accuracies"] = (
            reward_accuracies.cpu().numpy().tolist()
        )
        metrics[f"rewards_{train_test}/margins"] = (
            (pos_rewards - neg_rewards).cpu().numpy().tolist()
        )

        policy_neg_logps = all_gather_if_needed(
            policy_neg_logps.detach(), self.rank, self.world_size
        )
        metrics[f"logps_{train_test}/negative"] = (
            policy_neg_logps.cpu().numpy().tolist()
        )

        policy_pos_logps = all_gather_if_needed(
            policy_pos_logps.detach(), self.rank, self.world_size
        )
        metrics[f"logps_{train_test}/positive"] = (
            policy_pos_logps.cpu().numpy().tolist()
        )

        all_devices_losses = all_gather_if_needed(
            losses.detach(), self.rank, self.world_size
        )
        metrics[f"loss/{train_test}"] = (
            all_devices_losses.cpu().numpy().tolist()
        )

        if self.group_alphas or self.indiv_alphas:
            metrics[f"sparsity/l0_penalty_{train_test}"] = [
                grad_params["l0"].detach().item()
            ]
            metrics[f"sparsity/nonzero_params_{train_test}"] = [
                grad_params["nonzero_params"]
            ]
            metrics[f"sparsity/nonzero_z_{train_test}"] = [
                grad_params["nonzero_z"]
            ]
            metrics[f"sparsity/nonzero_groups_{train_test}"] = [
                grad_params["nonzero_groups"]
            ]
        return losses.mean(), grad_params, metrics

    def train_loop(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f"Using {self.config.optimizer} optimizer")

        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            [
                {"params": self.diff_params, "lr": self.config.lr},
                {
                    "params": self.alpha_params + self.alpha_groups,
                    "lr": self.config.sparse_lr,
                },
            ]
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            ),
        )

        # if len(self.alpha_params + self.alpha_groups) > 0:
        #    self.alpha_optimizer = getattr(torch.optim, self.config.optimizer)(
        #        self.alpha_params + self.alpha_groups,
        #        lr=self.config.sparse_lr,
        #    )
        #    self.alpha_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #        self.alpha_optimizer,
        #        lr_lambda=lambda step: min(
        #            1.0, (step + 1) / (self.config.warmup_steps + 1)
        #        ),
        #    )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        for batch in self.train_iterator:
            if self.example_counter % self.config.eval_every == 0 and (
                self.example_counter > 0 or self.config.do_first_eval
            ):
                result = self.eval()
                if result == -1:
                    break

            self.train(batch)

        self.eval_ppl()

    def train(self, batch):
        """
        Run single train step.
        """
        start_time = time.time()
        batch_metrics = defaultdict(list)
        for microbatch_idx in range(self.config.gradient_accumulation_steps):
            # batch:
            # {
            #   "pos_input_ids": Tensor[batch, seq],
            #   "pos_attention_mask": Tensor[batch, seq],
            #   "neg_input_ids": Tensor[batch, seq],
            #   "neg_attention_mask": Tensor[batch, seq],
            # }
            global_microbatch = slice_and_move_batch_for_device(
                batch,
                microbatch_idx,
                self.config.gradient_accumulation_steps,
                self.rank,
            )
            local_microbatch = slice_and_move_batch_for_device(
                global_microbatch, self.rank, self.world_size, self.rank
            )
            loss, sparse_grad_params, metrics = self.get_batch_metrics(
                local_microbatch, self.config.loss, train=True
            )
            # (loss / self.config.gradient_accumulation_steps).backward()

            (loss / self.config.gradient_accumulation_steps).backward()

            for k, v in metrics.items():
                batch_metrics[k].extend(v)

        self.set_grads(sparse_grad_params["grad_params"])

        # if self.indiv_alphas or self.group_alphas:
        #    l0_loss = self.sparse_alpha * sparse_grad_params["l0"]
        #    l0_loss.backward()

        if self.indiv_alphas:
            alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.alpha_params, self.config.max_grad_norm
            )
        if self.group_alphas:
            grouped_alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.alpha_groups, self.config.max_grad_norm
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.diff_params,
            self.config.max_grad_norm,
        ).item()

        if self.alpha_optimizer is not None:
            self.alpha_optimizer.step()
            self.alpha_optimizer.zero_grad()
        if self.alpha_scheduler is not None:
            self.alpha_scheduler.step()

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        step_time = time.time() - start_time
        examples_per_second = self.config.batch_size / step_time
        batch_metrics["examples_per_second"].append(examples_per_second)
        batch_metrics["grad_norm"].append(grad_norm)
        if self.indiv_alphas:
            batch_metrics["alpha_grad_norm"].append(alpha_grad_norm)
        if self.group_alphas:
            batch_metrics["grouped_alpha_grad_norm"].append(
                grouped_alpha_grad_norm
            )

        self.batch_counter += 1
        self.example_counter += self.config.batch_size

        if (
            self.last_log is None
            or time.time() - self.last_log
            > self.config.minimum_log_interval_secs
        ):
            mean_train_metrics = {
                k: sum(v) / len(v) for k, v in batch_metrics.items()
            }
            mean_train_metrics["counters/examples"] = self.example_counter
            mean_train_metrics["counters/updates"] = self.batch_counter
            rank0_print(
                f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
            )

            if self.config.wandb.enabled and self.rank == 0:
                wandb.log(mean_train_metrics, step=self.example_counter)

            self.last_log = time.time()

    def eval(self):
        """
        Run evaluation.
        """
        rank0_print(
            f"Running evaluation after {self.example_counter} train examples"
        )
        self.model.eval()

        standard_eval = self._eval()
        if (
            self.config.include_toxic
            and self.example_counter % self.config.eval_toxic_every == 0
        ):
            self._toxic_eval()
        return standard_eval

    def _eval(self):
        """
        Run evaluation.
        """
        all_eval_metrics = defaultdict(list)
        if self.config.sample_during_eval:
            all_policy_samples = []

        for eval_batch in (
            tqdm(self.eval_batches, desc="Computing eval metrics")
            if self.rank == 0
            else self.eval_batches
        ):

            local_eval_batch = slice_and_move_batch_for_device(
                eval_batch, self.rank, self.world_size, self.rank
            )
            with torch.no_grad():
                _, _, eval_metrics = self.get_batch_metrics(
                    local_eval_batch, self.config.loss, train=False
                )

            for k, v in eval_metrics.items():
                try:
                    all_eval_metrics[k].extend(v)
                except:
                    breakpoint()

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
                (policy_samples,) = self.get_batch_samples(local_eval_batch)

                all_policy_samples.extend(policy_samples)

            rank0_print("Policy samples:")
            rank0_print(json.dumps(all_policy_samples[:10], indent=2))

        mean_eval_metrics = {
            k: sum(v) / len(v) for k, v in all_eval_metrics.items()
        }
        self.val_metric_value = mean_eval_metrics[
            self.config.validation_metric
        ]

        rank0_print(
            f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
        )

        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(mean_eval_metrics, step=self.example_counter)

        if self.example_counter == 0:
            return 0

        if (
            self.val_metric_value is not None
            and self.val_metric_value * self.val_direction
            > self.val_direction * self.best_val_metric
        ):
            self.best_val_metric = self.val_metric_value

            rank0_print(
                f"\n=====\nNew best for {self.config.validation_metric}: {self.best_val_metric}.\n=====\n"
            )
            self.patience = 0

            if self.example_counter % self.config.eval_every == 0:
                if self.config.debug:
                    rank0_print("skipping save in debug mode")
                else:
                    output_dir = os.path.join(self.run_dir, "checkpoints")
                    rank0_print(
                        f"Creating checkpoint to write to {output_dir}..."
                    )
                    self.save(output_dir, mean_eval_metrics)
        else:
            self.patience += 1
            if self.patience >= self.config.validation_patience:
                rank0_print("Ran out of patience, stopping training...")
                return -1

        return 0

    def eval_ppl(self):
        """
        Run PPL
        """
        try:
            ckpt_dir = os.path.join(self.run_dir, "checkpoints")
            diff_file = os.path.join(ckpt_dir, "diffs.pt")
            diff_state_dict = torch.load(diff_file)["state"]

            if self.indiv_alphas:
                alphas = torch.load(os.path.join(ckpt_dir, "alpha.pt"))[
                    "state"
                ]
            if self.group_alphas:
                alpha_groups = torch.load(
                    os.path.join(ckpt_dir, "alpha_groups.pt")
                )["state"]
        except:
            breakpoint()

        try:
            for name, param in self.model.named_parameters():
                mask = torch.ones((1))
                if self.indiv_alphas:
                    _alpha = alphas[name]
                    mask = concrete_stretched(
                        _alpha, self.concrete_lower, self.concrete_upper
                    )
                param.data.copy_(
                    self.orig[name].to(param.device)
                    + mask.to(param.device)
                    * diff_state_dict[name].to(param.device)
                )
        except:
            breakpoint()

        try:
            self.model.eval()
            ppl = eval_perplexity(self.model, self.tokenizer)
            print(f"Perplexity: {ppl.item()}")
        except:
            breakpoint()

    def _toxic_eval(self):
        """
        Gather some metrics pertaining to wiki103.
        """
        counts = []
        mean_counts = []
        immediate_counts = []
        toxic_words = Counter(self.toxic_tokens.tolist())
        for eval_batch in (
            tqdm(
                self.toxic_data_batches,
                desc="Evaluating toxic generation rate..",
            )
            if self.rank == 0
            else self.toxic_data_batches
        ):
            local_eval_batch = slice_and_move_batch_for_device(
                eval_batch, self.rank, self.world_size, self.rank
            )

            ctx = lambda: (
                FSDP.summon_full_params(
                    self.model, writeback=False, recurse=False
                )
                if "FSDP" in self.config.trainer
                else contextlib.nullcontext()
            )
            with ctx():
                policy_output = self.model.generate(
                    local_eval_batch["prompt_input_ids"],
                    attention_mask=local_eval_batch["prompt_attention_mask"],
                    max_length=self.config.max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            output = policy_output[
                :, local_eval_batch["prompt_input_ids"].shape[1] :
            ]

            for idx in range(output.shape[0]):
                _output = Counter(output[idx].tolist())
                common = _output & toxic_words
                _count = sum(common.values())
                counts.append(_count)
                mean_counts.append(
                    _count
                    / (output[idx] != self.tokenizer.pad_token_id).sum().item()
                )

                _output = Counter([output[idx][0].item()])
                common = _output & toxic_words
                _count = sum(common.values())
                immediate_counts.append(_count)

            policy_output = pad_to_length(
                output,
                self.config.max_length,
                self.tokenizer.pad_token_id,
            )
            policy_output = all_gather_if_needed(
                policy_output, self.rank, self.world_size
            )
            policy_output_decoded = self.tokenizer.batch_decode(
                policy_output, skip_special_tokens=True
            )

            rank0_print("Policy samples:")
            rank0_print(json.dumps(policy_output_decoded[:10], indent=2))

        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(
                {"toxicity/toxic_rate": np.mean(counts)},
                step=self.example_counter,
            )
            wandb.log(
                {"toxicity/toxic_rate_per_utt": np.mean(mean_counts)},
                step=self.example_counter,
            )
            wandb.log(
                {"toxicity/immediate_toxic_rate": np.mean(immediate_counts)},
                step=self.example_counter,
            )

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(
            self.diff_params,
            self.config.max_grad_norm,
        ).item()

    def write_state_dict(
        self,
        step: int,
        state: Dict[str, torch.Tensor],
        metrics: Dict,
        filename: str,
        dir_name: Optional[str] = None,
    ):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f"LATEST")

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f"writing checkpoint to {output_path}...")
        torch.save(
            {
                "step_idx": step,
                "state": state,
                "metrics": metrics if metrics is not None else {},
            },
            output_path,
        )

    def save(
        self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None
    ):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.model.state_dict()

        self.write_state_dict(
            self.example_counter,
            self.diff_params_map,
            metrics,
            "diffs.pt",
            output_dir,
        )
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(
            self.example_counter,
            optimizer_state_dict,
            metrics,
            "optimizer.pt",
            output_dir,
        )
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(
            self.example_counter,
            scheduler_state_dict,
            metrics,
            "scheduler.pt",
            output_dir,
        )

        self.write_state_dict(
            self.example_counter,
            self.alpha_params_map,
            None,
            "alpha.pt",
            output_dir,
        )

        if self.group_alphas:
            self.write_state_dict(
                self.example_counter,
                self.alpha_groups_map,
                None,
                "alpha_groups.pt",
                output_dir,
            )
