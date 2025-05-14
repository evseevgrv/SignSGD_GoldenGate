import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from functools import partial

from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from peft_pretraining import training_utils
from galore_torch import AdamW, GaloreAdamW, CoordAdamW, BlockAdamW

def save_checkpoint(model, optimizer, scheduler, filename, world_size, rank):
    # Saving model
    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=False),
        FullOptimStateDictConfig(rank0_only=False),
    )
    model_state = model.state_dict()
    # dist.barrier()
    
    for r in range(world_size):
        dist.barrier()
        if r == rank:
            print(r)
            for p in model.parameters():
                print(optimizer.state[p])
            print()
        dist.barrier()
    # Saving optimizer
    optim_state = FSDP.optim_state_dict(model, optimizer)
    # dist.barrier()
    
    # Scheduler doesn't need special handling
    # scheduler_state = scheduler.state_dict()

    if rank == 0:
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': optim_state,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        }, f"{filename}")
    dist.barrier()


def load_checkpoint(model, optimizer, scheduler, filename, rank):
    # if rank == 0:
    checkpoint = torch.load(f"{filename}")
    # else:
    #     checkpoint = None
    
    # # Broadcast loaded state to all ranks
    # if int(os.environ["WORLD_SIZE"]) > 1:
    #     checkpoint = broadcast_object(checkpoint, src=0)
    
    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=False),
        FullOptimStateDictConfig(rank0_only=False),
    )

    # Loading model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Loading optimizer
    optim_state = FSDP.optim_state_dict_to_load(
        model, optimizer, checkpoint['optimizer_state_dict'],
    )
    optimizer.load_state_dict(optim_state)
    
    # Loading scheduler (normal way)
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    dist.barrier()

class SimpleModel(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.layer12 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.layer13 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, output_dim, bias=False)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer12(x)
        x = torch.relu(x)
        x = self.layer13(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def print_final_states(optimizer, world_size, rank, is_fsdp):
    if is_fsdp:
        dist.barrier()
        for r in range(world_size):
            if not is_fsdp and r:
                break
            dist.barrier()
            if rank == r:
                print(f"\nFinal optimizer states, worker {r}:")
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        print(f"Parameter size: {p.shape}")
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                print(f"State {k} size: {v.shape}")
            dist.barrier()
        dist.barrier()
    else:
        if rank == 0:
            print(f"\nFinal optimizer states, worker {rank}:")
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    print(f"Parameter size: {p.shape}")
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            print(f"State {k} size: {v.shape}")

def create_model(input_dim, world_size, rank, is_fsdp=False):
    model = SimpleModel(input_dim=input_dim).to(rank)
    if not is_fsdp:
        return model
    wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=0)
    dist.barrier()
    model = FSDP(
        model,
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=wrap_policy,
        # # use_orig_params=True
    )
    if rank==0:
        print(model)

    dist.barrier()
    for r in range(world_size):
        dist.barrier()
        if rank == r:
            for name, param in model.named_parameters():
                # Parameters should have roughly half the size on each GPU
                print(f"Rank {rank}, {name}, shape: {param.shape}, "
                    f"memory: {param.nelement() * param.element_size() / 1e6} MB")
        dist.barrier()
    dist.barrier()
    return model

def create_optimizer(model, optimizer_class, optimizer_kwargs, rank):
    params = [{'params': [], "is_proj_params": False}, {'params': [], "is_proj_params": True}]
    for i, p in enumerate(model.parameters()):
        # if not i:
        #     params[0]["params"].append(p)
        # else:
        params[1]["params"].append(p)
    optimizer = optimizer_class(params, **optimizer_kwargs)
    if rank == 0:
        for group in optimizer.param_groups:
            print(len(group["params"]), group["lr"])
    return optimizer

def create_scheduler(optimizer, n_steps):
    return training_utils.get_scheduler(
        optimizer=optimizer,
        scheduler_type="cosine",
        num_training_steps=n_steps,
        warmup_steps=int(n_steps * 0.1),
        min_lr_ratio=0.1,
        cycle_length=n_steps,
    )

def init(optimizer_class, optimizer_kwargs, world_size, rank, is_fsdp, is_sch, n_steps):
    torch.manual_seed(0)
    input_dim = 20
    batch_size = 64
    model = create_model(input_dim, world_size, rank, is_fsdp=is_fsdp)
    X, y = generate_data(batch_size, input_dim, rank)
    optimizer = create_optimizer(model, optimizer_class, optimizer_kwargs, rank)
    if is_sch:
        scheduler = create_scheduler(optimizer, n_steps)
    else:
        scheduler = None
    return model, optimizer, scheduler, X, y
    
def setup(world_size):
    local_rank = int(os.environ["LOCAL_RANK"])
    if dist.is_initialized():
        cleanup()
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    
def cleanup():
    dist.destroy_process_group()

def generate_data(batch_size, input_dim, device):
    # Generate synthetic regression data
    X = torch.randn(batch_size, input_dim, device=device)
    # True parameters
    w = torch.randn(input_dim, 1, device=device)
    # True relationship with some noise
    y = X @ w + 0.1 * torch.randn(batch_size, 1, device=device)
    return X, y

def train_step(model, optimizer, scheduler, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.item()

def train(model, optimizer, scheduler, X, y, rank, start, finish):
    losses = []
    for step in range(start, finish):
        loss = train_step(model, optimizer, scheduler, X, y)
        losses.append(loss)
        
        if rank == 0 and (step % 10 == 0 or step == finish - 1):
            print(f"Step {step}, Loss: {loss:.6f}")
    return losses

def save_single_gpu(model, optimizer, scheduler, filename, world_size=None, rank=None):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }, f"{filename}")

def load_single_gpu(model, optimizer, scheduler, filename, rank):
    ckpt = torch.load(f"{filename}")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

def test(optimizer_class, optimizer_kwargs, is_fsdp, is_sch, n_steps, saveload_strategy):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if is_fsdp:
        setup(world_size)
    model, optimizer, scheduler, X, y = init(optimizer_class, optimizer_kwargs, world_size, rank, is_fsdp=is_fsdp, is_sch=is_sch, n_steps=n_steps)
    losses = []
    if saveload_strategy != "no":
        filename = "fsdp_state.pt" if is_fsdp else "baseline_state.pt"
        if saveload_strategy == "saveload":
            losses += train(model, optimizer, scheduler, X, y, rank, 0, int(n_steps // 2))
            save_func = save_checkpoint if is_fsdp else save_single_gpu
            save_func(model, optimizer, scheduler, filename, world_size, rank)
            del model, optimizer, scheduler
            model, optimizer, scheduler, X, y = init(optimizer_class, optimizer_kwargs, world_size, rank, is_fsdp=is_fsdp, is_sch=is_sch, n_steps=n_steps)
        load_func = load_checkpoint if is_fsdp else load_single_gpu
        load_func(model, optimizer, scheduler, filename, rank)
    start = 0 if saveload_strategy == "no" else int(n_steps // 2)
    losses += train(model, optimizer, scheduler, X, y, rank, start, n_steps)
    print_final_states(optimizer, world_size, rank, is_fsdp=False)
    if is_fsdp:
        cleanup()
    return losses


# def sanity_check_baseline(optimizer_class, optimizer_kwargs, is_sch, n_steps):
#     rank = int(os.environ["LOCAL_RANK"])
#     world_size = int(os.environ["WORLD_SIZE"])
#     torch.manual_seed(0)
#     input_dim = 20
#     batch_size = 64
#     model = SimpleModel(input_dim=input_dim).to(rank)
#     X, y = generate_data(batch_size, input_dim, rank)
#     params = [{'params': [], "is_proj_params": False}, {'params': [], "is_proj_params": True}]
#     for i, p in enumerate(model.parameters()):
#         params[1]["params"].append(p)
#     optimizer = optimizer_class(params, **optimizer_kwargs)
#     if is_sch:
#         scheduler = training_utils.get_scheduler(
#             optimizer=optimizer,
#             scheduler_type="cosine",
#             num_training_steps=n_steps,
#             warmup_steps=int(n_steps * 0.1),
#             min_lr_ratio=0.1,
#             cycle_length=n_steps,
#         )
#     else:
#         scheduler = None
#     losses = []
#     for step in range(n_steps):
#         loss = train_step(model, optimizer, scheduler, X, y)
#         losses.append(loss)
        
#         if rank == 0 and (step % 10 == 0 or step == n_steps - 1):
#             print(f"Step {step}, Loss: {loss:.6f}")
#     return losses


def run_test(optimizer_class, optimizer_kwargs, world_size=2):
    losses_baseline = test(optimizer_class, optimizer_kwargs, is_fsdp=False, is_sch=True, n_steps=100, saveload_strategy="no")
    losses_baseline_saveload = test(optimizer_class, optimizer_kwargs, is_fsdp=False, is_sch=True, n_steps=100, saveload_strategy="saveload")
    assert torch.tensor(losses_baseline).allclose(torch.tensor(losses_baseline_saveload))
    assert torch.tensor(losses_baseline).eq(torch.tensor(losses_baseline_saveload)).all()

    losses_baseline_load = test(optimizer_class, optimizer_kwargs, is_fsdp=False, is_sch=True, n_steps=100, saveload_strategy="load")
    assert torch.tensor(losses_baseline[-50:]).allclose(torch.tensor(losses_baseline_load[-50:]))
    assert torch.tensor(losses_baseline[-50:]).eq(torch.tensor(losses_baseline_load[-50:])).all()

    # losses_fsdp_sizebased = test(optimizer_class, optimizer_kwargs, is_fsdp=True, is_sch=True, n_steps=100, saveload_strategy="no")
    # assert torch.tensor(losses_baseline).allclose(torch.tensor(losses_fsdp_sizebased))
    # assert torch.tensor(losses_baseline).eq(torch.tensor(losses_fsdp_sizebased)).all()

    # losses_fsdp_saveload = test(optimizer_class, optimizer_kwargs, is_fsdp=True, is_sch=True, n_steps=100, saveload_strategy="saveload")
    # assert torch.tensor(losses_baseline).allclose(torch.tensor(losses_fsdp_saveload))
    # assert torch.tensor(losses_baseline).eq(torch.tensor(losses_fsdp_saveload)).all()

    losses_fsdp_load = test(optimizer_class, optimizer_kwargs, is_fsdp=True, is_sch=True, n_steps=100, saveload_strategy="load")
    assert torch.tensor(losses_baseline[-50:]).allclose(torch.tensor(losses_fsdp_load[-50:]))
    assert torch.tensor(losses_baseline[-50:]).eq(torch.tensor(losses_fsdp_load[-50:])).all()

if __name__ == "__main__":
    # # Test with AdamW
    # print("Testing AdamW:")
    # run_test(
    #     torch.optim.AdamW,
    #     {'lr': 0.01, 'weight_decay': 0.01}
    # )

    # Test frugal
    print("Testing frugal:")
    run_test(
        BlockAdamW,
        {"proj_params_lr_scale": 1.0,
        "update_gap" : 9,
        "density": 0.25,
        "num_layers": 4,
        "block_order": "descending",
        "inactive_update_rule": "no"}
    )