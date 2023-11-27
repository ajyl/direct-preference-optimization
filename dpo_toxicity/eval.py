"""
Train script
"""
from typing import Optional, Set

import os
import json
import socket
import resource
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import transformers
import hydra
from omegaconf import OmegaConf, DictConfig

import dpo_toxicity.evaluators as evaluators
from dpo_utils import (
    get_local_dir,
    get_local_run_dir,
    disable_dropout,
    init_distributed,
    get_open_port,
)

torch.backends.cuda.matmul.allow_tf32 = True
OmegaConf.register_new_resolver(
    "get_local_run_dir",
    lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs),
)


def worker_main(
    rank: int,
    world_size: int,
    config: DictConfig,
    policy: nn.Module,
    reference_model: Optional[nn.Module] = None,
):
    """
    Main function for each worker process
    (may be only 1 for BasicTrainer/TensorParallelTrainer).
    """
    if "FSDP" in config.evaluator:
        init_distributed(rank, world_size, port=config.fsdp_port)

    EvalClass = getattr(evaluators, config.evaluator)
    print(f"Creating evaluators on process {rank} with world size {world_size}")
    evaluator = EvalClass(
        policy,
        config,
        config.seed,
        config.local_run_dir,
        reference_model=reference_model,
        rank=rank,
        world_size=world_size,
    )

    evaluator.eval()


@hydra.main(version_base=None, config_path="config", config_name="eval")
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

    if "FSDP" in config.evaluator and config.fsdp_port is None:
        free_port = get_open_port()
        print("no FSDP port specified; using open port for FSDP:", free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)

    print("=" * 80)
    print(f"Writing to {socket.gethostname()}:{config.local_run_dir}")
    print("=" * 80)

    os.environ["XDG_CACHE_HOME"] = get_local_dir(config.local_dirs)
    print("building policy")
    model_kwargs = (
        {"device_map": "balanced"} if config.evaluator == "BasicEvaluator" else {}
    )
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=get_local_dir(config.local_dirs),
        low_cpu_mem_usage=True,
        torch_dtype=policy_dtype,
        **model_kwargs,
    )
    disable_dropout(policy)

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

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location="cpu")
        step, metrics = state_dict["step_idx"], state_dict["metrics"]
        print(
            f"loading pre-trained weights at step {step} from \
            {config.model.archive} with metrics \
            {json.dumps(metrics, indent=2)}"
        )
        policy.load_state_dict(state_dict["state"])
        reference_model.load_state_dict(state_dict["state"])

        print("loaded pre-trained weights")

    if "FSDP" in config.evaluator:
        world_size = torch.cuda.device_count()
        print("starting", world_size, "processes for FSDP training")
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"setting RLIMIT_NOFILE soft limit to {hard} from {soft}")
        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, config, policy, reference_model),
            join=True,
        )
    else:
        print("starting single-process worker")
        worker_main(0, 1, config, policy, reference_model)


if __name__ == "__main__":
    main()
