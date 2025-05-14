import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from functools import partial

from torch import autocast

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from typing import List

from utils import training_utils, args_utils
from utils.dataloader import PreprocessedIterableDataset
from utils.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer

from frugal import prepare_proj_params

from nirvana_utils import copy_out_to_snapshot, copy_snapshot_to_out
from utils import compile_save_load_utils
import gc

from safetensors.torch import load_model

import functools

transformers.logging.set_verbosity_error()

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "constant", "cosine", "cosine_restarts", "rounded"])
    parser.add_argument("--scheduler_cycle_length", type=int, default=None)
    parser.add_argument("--scheduler_min_power", type=int, default=-20)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", default=False, action="store_true")
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--general_save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_dir_prefix", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16" if torch.cuda.is_bf16_supported() else "fp32")
    parser.add_argument('--amp', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    parser.add_argument("--wandb_tags", type=List, default=[])
    parser.add_argument("--wandb_name_prefix", type=str, default=None)
    parser.add_argument("--run_final_eval", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--run_old_eval", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--final_save", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug_train_data", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug_print", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])

    parser.add_argument("--fsdp", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cpu_offload", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--mp_policy_param", type=str, default="float32")
    parser.add_argument("--mp_policy_reduce", type=str, default="float32")
    parser.add_argument("--mp_policy_buffer", type=str, default="float32")
    parser.add_argument("--use_orig_params", default=False, action=argparse.BooleanOptionalAction)

    # Proj parameters
    parser.add_argument("--proj_params_lr_scale", type=float, default=1.0)
    parser.add_argument("--update_gap", type=int, default=50)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument('--reset_statistics', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--inactive_update_rule", type=str, default="sign_sgd", choices=["no", "sgd", "sign_sgd"])
    parser.add_argument("--inactive_lr_scale", type=float, default=1.0)
    parser.add_argument("--proj_norms", default=False, action=argparse.BooleanOptionalAction) 
    parser.add_argument("--proj_embeds", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--proj_logits", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--rank", type=int, default=0)

    # Galore parameters
    parser.add_argument("--proj_side", type=str, default="std", choices=["std", "reverse_std", "right", "left", "full"])
    parser.add_argument("--proj_type", type=str, default="svd", choices=["svd", "random", "randperm", "power_iteration"])

    # Coord parameters
    parser.add_argument("--coord_choice", type=str, default="columns", choices=["columns", "rows", "randk"])

    # Block parameters
    parser.add_argument("--block_order", type=str, default="random", choices=['random', 'ascending', 'descending', 'mirror'])

    # APOLLO parameters
    parser.add_argument("--apollo_proj", type=str, default="random") # "random" or "svd"
    parser.add_argument("--apollo_scale_type", type=str, default="tensor") # "tensor" or "channel"
    parser.add_argument("--apollo_scale", type=float, default=1.0) # scale for gradient scaling factor
    parser.add_argument("--apollo_scale_front", action='store_true') # put the nl before or after scale the gradient with the apollo_scale

    # LDAdam parameters
    parser.add_argument("--ldadam_rho", type=float, default=0.908)
    parser.add_argument("--ldadam_proj_method", type=str, default="power_iteration")
    parser.add_argument("--ldadam_error_feedback", default=False, action=argparse.BooleanOptionalAction)

    # Fira parameters
    parser.add_argument("--fira_alpha", type=float, default=1.0)

    # Delayed AdamW parameters
    parser.add_argument("--delay_start_step", type=int, default=10000)
    parser.add_argument("--grad_taylor_approx", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--taylor_linear_coef", type=float, default=1.0)

    # Metrics
    parser.add_argument("--measure_time", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--collect_grads", default=False, action="store_true")

    # Optimization parameters
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1_000)

    # Adam parameters
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--eps", type=float, default=1e-8)
    # SGD parameters
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dampening", type=float, default=0)
    parser.add_argument("--sgd_sign_update", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--sign_norm", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--normalized", default=False, action=argparse.BooleanOptionalAction)
    
    # aid signsgd
    parser.add_argument("--l_inf", type=float, default=None)
    parser.add_argument("--d_0", type=float, default=None)
    parser.add_argument("--lower_bound", type=float, default=None)
    parser.add_argument("--clamp_level", type=float, default=None)

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    parser.add_argument("--local_train_data", default=False, action="store_true")
    parser.add_argument('--streaming', default=True, action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args

from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

def save_checkpoint_fsdp(model, optimizer, scheduler, filename, world_size, rank):
    # Saving model
    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=False),
        FullOptimStateDictConfig(rank0_only=False),
    )
    model_state = model.state_dict()
    optim_state = FSDP.optim_state_dict(model, optimizer)
    if rank == 0:
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': optim_state,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        }, f"{filename}")
    dist.barrier()

def load_checkpoint_fsdp(model, optimizer, scheduler, filename, rank, compile):
    checkpoint = torch.load(f"{filename}")
    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=False),
        FullOptimStateDictConfig(rank0_only=False),
    )
    if compile:
        checkpoint['model_state_dict'] = compile_save_load_utils.transform_state_dict_keys(checkpoint['model_state_dict'])
        checkpoint['optimizer_state_dict'] = compile_save_load_utils.transform_optimizer_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optim_state = FSDP.optim_state_dict_to_load(
        model, optimizer, checkpoint['optimizer_state_dict'],
    )
    optimizer.load_state_dict(optim_state)
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    dist.barrier()

def save_checkpoint(args, model, optimizer, scheduler, global_rank, world_size, update_step, global_step, 
                    run_config, tokens_seen, tokens_seen_before, update_time, seed_for_shuffle):
    current_model_directory = f"{os.path.join(args.general_save_dir, args.save_dir)}"
    os.makedirs(os.path.join(args.general_save_dir, args.save_dir), exist_ok=True)
    if args.fsdp:
        dist.barrier()
        save_checkpoint_fsdp(model, optimizer, scheduler, f"{current_model_directory}/checkpoint.pt", world_size, global_rank)
        dist.barrier()
    elif global_rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        model_state_dict = model.state_dict()
        # if compile:

        torch.save(model_state_dict, f"{current_model_directory}/pytorch_model.bin")

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir if args.wandb else None,
            "dtype": args.dtype,
        }
        if args.optimizer == "badam":
            optimizer_checkpoint["active_param_prefixs"] = optimizer.active_param_prefixs
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")
    if global_rank == 0:
        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
            "seed_for_shuffle": seed_for_shuffle,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)
            
        # save wandb related info
        wandb_info = {
            "wandb_id": wandb.run.id if args.wandb else "",
        }
        with open(f"{os.path.join(args.general_save_dir, args.save_dir)}/wandb.json", "w") as f:
            json.dump(wandb_info, f, indent=4)

        copy_out_to_snapshot(args.general_save_dir)
    dist.barrier()

@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size, 
                   args_dtype, args_amp, short_debug_eval=False):
    _time = time.time()
    
    if "INPUT_PATH" in os.environ or "c4_val" in os.listdir(os.getcwd()):
        input_path = os.environ["INPUT_PATH"] if "INPUT_PATH" in os.environ else os.getcwd()
        data_files = {"val": [os.path.join(input_path, "c4_val/*.json.gz")]}
        val_data = datasets.load_dataset('json', data_files=data_files, split='val', streaming=args.streaming)
    else:
        val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)

    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 0
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    max_batches = int(200 / int(os.environ["WORLD_SIZE"]) * (batch_size / 256)) + 1 if not short_debug_eval else 3
    for batch in val_data_mapped.batch(batch_size=batch_size):
        if total_batches == max_batches:
            break

        total_batches += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        with autocast(device_type='cuda', dtype=torch.bfloat16 if not args_dtype == torch.float16 else torch.float16, enabled=args_amp):
            loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    stable_rank_dict = None

    return total_loss, evaluated_on_tokens, stable_rank_dict


@torch.no_grad()
def evaluate_model_old(model, preprocess_batched, pad_idx, global_rank, world_size, device, args):
    _time = time.time()
    if "INPUT_PATH" in os.environ or "c4_val" in os.listdir(os.getcwd()):
        input_path = os.environ["INPUT_PATH"] if "INPUT_PATH" in os.environ else os.getcwd()
        data_files = {"val": [os.path.join(input_path, "c4_val/*.json.gz")]}
        val_data = datasets.load_dataset('json', data_files=data_files, split='val', streaming=True)
    else:
        val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)
    # val_data = datasets.load_from_disk('/fsx-storygen/yuandong/c4_processed/val')

    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")
    batch_size = args.batch_size

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        with autocast(device_type='cuda', dtype=torch.bfloat16 if not args.dtype == torch.float16 else torch.float16, enabled=args.amp and (args.attn_implementation == "flash_attention_2")):
            loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens

def main(args):
    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.debug:
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
        torch.use_deterministic_algorithms(True)

    # starting DDP enviroment
    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    
    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"
    
    # calculating per device batch_size
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"
    # turn off logger
    if global_rank != 0: logger.remove()

    # args    
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # model
    model_config = AutoConfig.from_pretrained(args.model_config)
    args.hidden_size = model_config.hidden_size
    if not args.use_hf_model:
        model = LlamaForCausalLM(model_config)
    else:
        model = AutoModelForCausalLM.from_config(model_config, attn_implementation=args.attn_implementation)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
    
    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    seed_for_shuffle = 42
    
    model = model.to(dtype=args.dtype)
    # model dtype
    if not args.fsdp:
        model = model.to(device=device)
        if not args.single_gpu:
            model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                # find_unused_parameters=True if args.optimizer == "badam" else False,
            )
    else:
        if global_rank == 0:
            print("MP policy")
            print(args.mp_policy_param)
            print(args.mp_policy_reduce)
            print(args.mp_policy_buffer)
        mp_policy = MixedPrecision(
            param_dtype=args.mp_policy_param,
            reduce_dtype=args.mp_policy_reduce,
            buffer_dtype=args.mp_policy_buffer,
        )
        # wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=500)
        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,
            },
        )
        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # or other strategies
            device_id=torch.cuda.current_device(),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            cpu_offload=CPUOffload(offload_params=True) if args.cpu_offload else None,  # Enable if needed
            use_orig_params=args.use_orig_params
        )
        if global_rank == 0:
            for name, p in model.named_parameters():
                print(name, p.shape)
        dist.barrier()
    # loading from checkpoint
    if global_rank == 0:
        print(model)
        print("Model dtype: ", next(model.parameters()).dtype)
    dist.barrier()
    if local_rank == 0:
        if not os.path.isdir(args.general_save_dir):
            copy_snapshot_to_out(args.general_save_dir)  # Yandex Nirvana
    dist.barrier()
    resume_from_checkpoint = os.path.exists(os.path.join(args.general_save_dir, args.save_dir, "model.safetensors")) or os.path.exists(os.path.join(args.general_save_dir, args.save_dir, "pytorch_model.bin")) or os.path.exists(os.path.join(args.general_save_dir, args.save_dir, "checkpoint.pt"))
    # resume_from_checkpoint = False
    dist.barrier()

    if resume_from_checkpoint:
        logger.info("*" * 40)
        with open(os.path.join(args.general_save_dir, args.save_dir, "wandb.json")) as f:
            wandb_id = json.load(f)["wandb_id"]
            wandb_resume = True
            # wandb_id = None
            # wandb_resume = None
        if os.path.exists(os.path.join(args.general_save_dir, args.save_dir, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {os.path.join(args.general_save_dir, args.save_dir)}")
            with open(os.path.join(args.general_save_dir, args.save_dir, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            seed_for_shuffle = _old_state["seed_for_shuffle"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {os.path.join(args.general_save_dir, args.save_dir)}, global step will start from zero")
        logger.info("*" * 40)
    else:
        os.makedirs(os.path.join(args.general_save_dir, args.save_dir), exist_ok=True)
        wandb_id = None
        wandb_resume = None
        optimizer_scheduler_state_dict = None

    # train data
    if args.local_train_data:
        if "INPUT_PATH" in os.environ:
            if not args.debug_train_data:
                data_files = {"train": [f"{os.environ['INPUT_PATH']}/*json.gz"]}
            else:
                data_files = {"train": [f"{os.environ['INPUT_PATH']}/c4_val/*json.gz"]}
        else:
            data_files = {"train": [f"{os.getcwd()}/c4_val/*json.gz"]}
        data = datasets.load_dataset('json', data_files=data_files, split='train', streaming=args.streaming)
    else:
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers, pin_memory=True, persistent_workers=True)

    # initialize wandb
    if global_rank == 0 and args.wandb:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "frugal"),
            name=args.wandb_name,
            tags=args.wandb_tags,
            id=wandb_id,
            resume=wandb_resume,
        )

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        if args.wandb:
            wandb.config.update(run_config, allow_val_change=True)
            wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)


    param_groups = prepare_proj_params(model, proj_norms=args.proj_norms, proj_embeds=args.proj_embeds,
                                       proj_logits=args.proj_logits, fsdp=args.fsdp, use_orig_params=args.use_orig_params)
    if global_rank == 0:
        print(f"proj_norms: {args.proj_norms}, proj_embeds: {args.proj_embeds}, proj_logits: {args.proj_logits}, \nNumber of proj_params: {len(param_groups[1]['params'])}, number of NON proj_params: {len(param_groups[0]['params'])}")

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    for group in param_groups:
        if group["is_proj_params"]:
            logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in group['params'] ) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {os.path.join(args.general_save_dir, args.save_dir)} every {args.save_every} update steps")

    # creating optimizer
    optimizer = training_utils.get_optimizer(param_groups, args, model)
    if global_rank == 0:
        print(optimizer)
        for i, group in enumerate(optimizer.param_groups):
            print(i, group["is_proj_params"], len(group["params"]))

    # creating scheduler
    scheduler = training_utils.get_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        cycle_length=args.scheduler_cycle_length,
    )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    local_step = 0  # when save_dir is used, local_step != global_step

    # loading optimizer and scheduler states from checkpoint
    if resume_from_checkpoint:
        current_model_directory = f"{os.path.join(args.general_save_dir, args.save_dir)}"
        logger.info(f"Loading model from {current_model_directory}")
        try:
            if args.optimizer == "badam":
                retries = 0
                assert len(optimizer_scheduler_state_dict["active_param_prefixs"]) ==  len(optimizer.active_param_prefixs)
                while not all(x == y for x, y in zip(optimizer_scheduler_state_dict["active_param_prefixs"], optimizer.active_param_prefixs)):
                    optimizer.switch_trainable_params()
                    retries += 1
                    if retries == 1000:
                        raise ValueError("Broken checkpoint.")
        except:
            import warnings
            warnings.warn('You are resuming training from checkpoint but reinitializing optimizer and scheduler.')
        
        if not args.fsdp:
            if os.path.exists(os.path.join(args.general_save_dir, args.save_dir, "model.safetensors")):
                load_model(model, os.path.join(args.general_save_dir, args.save_dir, "model.safetensors"))
            else:
                checkpoint_path = os.path.join(args.general_save_dir, args.save_dir, "pytorch_model.bin")
                model_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
                if args.compile:
                    model_state_dict = compile_save_load_utils.transform_state_dict_keys(model_state_dict)
                model.load_state_dict(model_state_dict, strict=True)
                
            optimizer_scheduler_state_dict = torch.load(os.path.join(args.general_save_dir, args.save_dir, "optimizer.pt"), map_location=device)
            logger.info(f"Model successfully loaded (strict=True policy)")
            optimizer.load_state_dict(optimizer_scheduler_state_dict["optimizer"])
            scheduler.load_state_dict(optimizer_scheduler_state_dict["scheduler"])
        else:
            dist.barrier()
            load_checkpoint_fsdp(model, optimizer, scheduler, f"{current_model_directory}/checkpoint.pt", global_rank, args.compile)
            dist.barrier()
            # for r in range(int(os.environ["WORLD_SIZE"])):
            #     if dist.get_rank() == r:
            #         print(r, optimizer.state[next(model.parameters())])
            #     dist.barrier()
            # dist.barrier()

    start_dataloader_time = time.time()
    update_time = time.time()
    # # 
    # skip_mode is for "honest" loading from checkpoint
    # since we need to preserve data order.
    # unfortunately, there is no way to skip first N steps for
    # streaming dataset, so it is done in stupid loop.
    # however, it is take too much time, so for now it is disables
    skip_mode = False # if not global_step else True
    skip_cnt = 0
    stop_flag = False
    
    if args.collect_grads:
        for group in param_groups:
            if not group.get("is_proj_params", False):
                continue
            grad_dicts = [{} for param in group["params"]]

    if args.compile:
        model = torch.compile(model)

    # TRAINING LOOP
    while not stop_flag:
        for batch_idx, batch in enumerate(dataloader):
            
            torch.cuda.synchronize()
            step_start_time = time.time()

            # this code is for "honest" loading from checkpoint 
            # since we need to preserve data order.
            # unfortunately, there is no way to skip first N steps for
            # streaming dataset, so it is done in stupid loop.
            # however, it is take too much time, so for now it is disables
            if skip_mode:
                if not skip_cnt % 100:
                    print(skip_cnt)
                skip_cnt += 1
                skip_mode = skip_cnt < global_step
                if not skip_mode:
                    print(f"Skipping data total time: {time.time()-start_dataloader_time:.2f}")
                continue
            if update_step >= args.num_training_steps:
                logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
                print(f"Rank {global_rank} stopping training.")
                stop_flag = True
                break

            global_step += 1
            local_step += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

            # if args.optimizer in ["aid_sign_sgd", "aid_with_adam"] and update_step > 0:
            #      cur_params = {p: p.clone().detach() for group in optimizer.param_groups for p in group["params"]}
            #      prev_params = {p: optimizer.state[p]["prev_param"].clone().detach() for group in optimizer.param_groups for p in group["params"]}
            #      with torch.no_grad():
            #          for p in prev_params:
            #              p.copy_(prev_params[p])
            #      with autocast(device_type='cuda', dtype=torch.bfloat16 if not args.dtype == torch.float16 else torch.float16, enabled=args.amp):
            #          loss_prev = model(**batch, labels=labels).loss
            #      loss_prev.backward()
            #      custom_grad = {p: p.grad.clone() for group in optimizer.param_groups for p in group["params"]}
            #      with torch.no_grad():
            #          for p in prev_params:
            #              p.copy_(cur_params[p])

            # forward-backward
            with autocast(device_type='cuda', dtype=torch.bfloat16 if not args.dtype == torch.float16 else torch.float16, enabled=args.amp):
                loss = model(**batch, labels=labels).loss
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            if global_step % args.gradient_accumulation != 0:
                continue

            # The below code is only executed during the update step
            
            if not args.fsdp:
                total_norm = None
                norms = [torch.linalg.vector_norm(p.grad.data, 2) for p in model.parameters()]
                total_norm = torch.linalg.vector_norm(
                    torch.stack([norm for norm in norms]), 2
                )
                if global_rank == 0 and args.wandb:
                    wandb.log({
                        "grad_norm": total_norm.item(),
                        },
                        step=update_step,
                    )
            # add grad clipping
            if args.grad_clipping != 0.0: 
                # TODO fp16 amp
                # if args.dtype == torch.float16:
                #     scaler.unscale_(optimizer)
                if not args.fsdp:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
                else:
                    model.clip_grad_norm_(args.grad_clipping, norm_type=2)

            if global_rank == 0: pbar.update(1)
            
            if args.collect_grads:
                if global_rank == 0:
                    if ((0 <= update_step <= 100) or 
                        (100 <= update_step <= 10000 and not update_step % 100) or 
                        (10000 <= update_step < 100000 and not update_step % 1000)):
                        for group in param_groups:
                            if not group.get("is_proj_params", False):
                                continue
                            for p, grad_dict in zip(group["params"], grad_dicts):
                                grad_dict[update_step] = p.grad.cpu()
                if update_step == 100:
                    if global_rank == 0:
                        current_model_directory = f"{os.path.join(args.general_save_dir, args.save_dir)}/"
                        torch.save(grad_dicts, f"{current_model_directory}/grad_dicts.pt")
                    return

            closure = None
            if args.optimizer in ["aid_sign_sgd", "aid_with_adam"] and update_step == 0:
                gathered_losses = [torch.zeros_like(loss) for _ in range(world_size)]
                dist.all_gather(gathered_losses, loss)
                gathered_loss = sum([t.item() for t in gathered_losses]) / world_size
                closure = functools.partial(optimizer._get_loss, loss=gathered_loss)

            # if args.optimizer in ["aid_sign_sgd", "aid_with_adam"] and update_step > args.warmup_steps:
            #      for group in optimizer.param_groups:
            #          for p in group["params"]:
            #              optimizer.state[p]["prev_grad"] = custom_grad[p].clone()

            optimizer.step(closure)
            
            if global_rank == 0 and args.debug_print:
                p = next(model.parameters())
                print(f"Weights: {p.data.dtype}\nGrads: {p.grad.data.dtype}\nOptimizer state: {optimizer.state[p]['exp_avg'].dtype}")
            scheduler.step()
            optimizer.zero_grad()

            update_step += 1
            update_time = time.time() - update_time

            # evaluation
            if update_step % args.eval_every == 0:
                logger.info(f"Performing evaluation at step {update_step}")
                total_loss, evaluated_on_tokens, stable_rank_dict = evaluate_model(
                    model, preprocess_batched, pad_idx, global_rank, world_size, device, args.eval_batch_size, 
                    args_dtype=args.dtype, args_amp=args.amp, short_debug_eval=args.debug
                )
                if global_rank == 0 and args.wandb:
                    wandb.log({
                        "final_eval_loss": total_loss,
                        "final_eval_tokens": evaluated_on_tokens,
                        },
                        step=update_step,
                    )
                if args.run_old_eval:
                    total_loss_old, evaluated_on_tokens = evaluate_model_old(model, preprocess_batched, pad_idx, global_rank, world_size, device, args)
                    if global_rank == 0 and args.wandb:
                        wandb.log({
                            "final_eval_loss_old": total_loss_old,
                            },
                            step=update_step,
                        )
                torch.cuda.empty_cache()
                logger.info(f"Eval loss at step {update_step}: {total_loss}")

            # save checkpoint by save_every
            if update_step % args.save_every == 0:
                current_model_directory = f"{os.path.join(args.general_save_dir, args.save_dir)}"
                logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
                save_checkpoint(args, model, optimizer, scheduler, global_rank, world_size, update_step, global_step, 
                    run_config, tokens_seen, tokens_seen_before, update_time, seed_for_shuffle)

            if args.optimizer in ["aid_sign_sgd", "aid_with_adam"] and update_step > args.warmup_steps:
                for group in optimizer.param_groups:
                    if "prev_gamma" in group:
                        # group["lr"] = group["prev_gamma"].clone()
                        last_gamma = group["prev_gamma"][-1]
                        group["lr"] = last_gamma.clone() if isinstance(last_gamma, torch.Tensor) else torch.tensor(last_gamma)
            lr = optimizer.param_groups[-1]["lr"]
            tokens_in_update = tokens_seen - tokens_seen_before
            tokens_seen_before = tokens_seen
            batches_in_update = args.gradient_accumulation * world_size
            
            if global_rank == 0 and args.wandb:
                wandb.log({
                    "loss": loss.item(),
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                    "throughput_tokens": tokens_in_update / update_time,
                    "throughput_examples": args.total_batch_size / update_time,
                    "throughput_batches": batches_in_update / update_time,
                    
                    },
                    step=update_step,
                )
            update_time = time.time()
            if update_step % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()


            torch.cuda.synchronize()
            step_time = time.time() - step_start_time

            if world_size > 1:
                step_time_tensor = torch.tensor([step_time], device=device)
                dist.all_reduce(step_time_tensor, op=dist.ReduceOp.MAX)
                max_step_time = step_time_tensor.item()
            else:
                max_step_time = step_time
            if global_rank == 0 and args.wandb:
                wandb.log({
                    "epoch_step_time": max_step_time,
                    "epoch_step_time_local": step_time,
                }, step=update_step)

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()
    
    # Final evaluation
    if args.run_final_eval:
        logger.info("Running final evaluation")
        model.eval()
        total_loss, evaluated_on_tokens, stable_rank_dict = evaluate_model(
            model, preprocess_batched, pad_idx, global_rank, world_size, device, args.eval_batch_size, 
            args_dtype=args.dtype, args_amp=args.amp, short_debug_eval=args.debug
        )
        if global_rank == 0 and args.wandb:
            wandb.log({
                "final_eval_loss": total_loss,
                "final_eval_tokens": evaluated_on_tokens,
                },
                step=update_step,
            )
            logger.info(f"Final eval loss: {total_loss}")
        if args.run_old_eval:
            total_loss_old, evaluated_on_tokens = evaluate_model_old(model, preprocess_batched, pad_idx, global_rank, world_size, device, args)
            if global_rank == 0 and args.wandb:
                wandb.log({
                    "final_eval_loss_old": total_loss_old,
                    },
                    step=update_step,
                )
            

    # Saving checkpoint
    current_model_directory = f"{os.path.join(args.general_save_dir, args.save_dir)}"
    if args.final_save:
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        save_checkpoint(args, model, optimizer, scheduler, global_rank, world_size, update_step, global_step, 
                    run_config, tokens_seen, tokens_seen_before, update_time, seed_for_shuffle)
    dist.destroy_process_group()
    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)