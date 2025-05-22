import os
import json
import time
import argparse
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import wandb

from utils import (
    set_seed,
    get_device,
    get_resnet18,
    get_simple_model,
    Logger,
    get_data_loaders,
    loader_to_device,
    top_k_accuracy,
    get_warmup_cosine_scheduler
)

NCOLS = 100

class BaseOptimizer:
    def __init__(
        self, model, data_loaders, loss_fn, device,
        lambda_value=None, scheduler=None, scheduler_config=None
    ):
        self.model = model
        self.train_data_loader = data_loaders[0]
        self.test_data_loader = data_loaders[1]
        self.train_batch_count = len(self.train_data_loader)
        self.test_batch_count = len(self.test_data_loader)
        self.loss_fn = loss_fn
        self.train_logger = Logger()
        self.test_logger = Logger()
        self.grads_epochs_computed = 0
        self.regularizer = lambda_value
        self.device = device
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {}
        self.loggers = {}

    def run(self, epochs, lr, exp_name=None, scheduler_func=None):
        if exp_name is None:
            exp_name = f"{self.__class__.__name__}_lr={lr}_epochs={epochs}"
            if self.regularizer is not None:
                exp_name += f"_lambda={self.regularizer}"
        config = {
            "method": self.__class__.__name__,
            "lr": lr,
            "epochs": epochs,
            "lambda": self.regularizer,
            "batch_size": BATCH_SIZE,
            "model": "swin_tiny"
        }
        wandb.init(
            project="vit-experiments",
            name=exp_name,
            config=config,
            tags=[args.method, "tiny-imagenet"]
        )
        if scheduler_func is not None:
            sched_cfg = {
                'warmup_epochs': 5,
                'total_epochs': epochs,
                'base_lr': lr,
                'warmup_lr': lr * 0.1,
                'min_lr': 1e-5,
                **self.scheduler_config
            }
            self.scheduler = scheduler_func(**sched_cfg)
        else:
            self.scheduler = lambda e: lr

        self.start_time = time.time()
        for epoch in tqdm(range(epochs), desc=exp_name, ncols=100):
            curr_lr = self.scheduler(epoch)
            self._train_epoch(curr_lr, epoch)
            self._test_epoch(epoch)

            tqdm.write(
                f"Train Loss: {self.train_logger.loss[-1]:.4f}, "
                f"Train Acc: {self.train_logger.accuracy[-1]:.4f}"
            )
            tqdm.write(
                f"Test Loss: {self.test_logger.loss[-1]:.4f}, "
                f"Test Acc: {self.test_logger.accuracy[-1]:.4f}"
            )
        self.loggers[exp_name] = {
            'train': self.train_logger.to_dict(),
            'test': self.test_logger.to_dict(),
        }
        wandb.finish()

    def _forward_backward(self, model, inputs, targets, zero_grad=True, is_test=False):
        if zero_grad:
            model.zero_grad()
        model.eval() if is_test else model.train()
        outputs = model(inputs)
        bs, num_logits = outputs.shape
        tmin, tmax = targets.min().item(), targets.max().item()
        if tmin < 0 or tmax >= num_logits:
            raise RuntimeError(
                f"Неправильные метки: targets в [{tmin},{tmax}], outputs.shape={outputs.shape}"
            )
        loss = self.loss_fn(outputs, targets)
        if self.regularizer is not None:
            for p in model.parameters():
                loss += (self.regularizer / 2) * torch.sum(p ** 2)
        if not is_test:
            loss.backward()
            self.grads_epochs_computed += 1 / self.train_batch_count
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        acc = correct / targets.size(0)
        topk = top_k_accuracy(outputs, targets, ks=(1, 2, 3, 4, 5))
        return loss, acc, topk

    def _test_epoch(self, epoch):
        total_loss = 0.0
        total_acc = 0.0
        topk_sum = {f'top@{k}': 0.0 for k in range(1, 6)}

        for inputs, targets in tqdm(
            loader_to_device(self.test_data_loader, self.device),
            desc='Testing', ncols=100, leave=False
        ):
            loss, acc, topk = self._forward_backward(
                self.model,
                inputs.to(self.device),
                targets.to(self.device),
                zero_grad=True,
                is_test=True
            )
            total_loss += loss.item()
            total_acc += acc
            for k, v in topk.items():
                topk_sum[k] += v

        avg_loss = total_loss / self.test_batch_count
        avg_acc = total_acc / self.test_batch_count
        avg_topk = {k: v / self.test_batch_count for k, v in topk_sum.items()}

        self.test_logger.append(
            avg_loss,
            avg_acc,
            self.grads_epochs_computed,
            time.time() - self.start_time,
            avg_topk
        )

        test_step = (epoch + 1) * self.train_batch_count
        wandb.log({
            "test/loss": avg_loss,
            "test/accuracy": avg_acc,
            "test/top1_acc": avg_topk['top@1'],
            "test/top5_acc": avg_topk['top@5'],
            "epoch": epoch
        }, step=test_step)

class AdamLike(BaseOptimizer):
    def __init__(
        self, model, data_loaders, loss_fn, device,
        lambda_value=None, scheduler=None, scheduler_config=None,
        beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
        head_lr=0.001, head_wd=0.001, num_last_layers=0,
        grad_clip=1.0
    ):
        super().__init__(model, data_loaders, loss_fn, device,
                         lambda_value, scheduler, scheduler_config)
        self.grad_clip = grad_clip
        head_mod = self.model.head.fc
        head_params = list(head_mod.parameters())
        last_layers = self.model.layers[-num_last_layers:]
        last_params = []
        for layer in last_layers:
            last_params += [p for p in layer.parameters()]
        self.adamw_params = head_params + last_params
        adamw_ids = {id(p) for p in self.adamw_params}
        self.body_params = [p for p in self.model.parameters()
                            if p.requires_grad and id(p) not in adamw_ids]

        self.opt_adamw = torch.optim.AdamW(
            self.adamw_params, lr=head_lr, weight_decay=head_wd
        )
        self.head_lr = head_lr

        self.beta1, self.beta2, self.eps, self.wd = beta1, beta2, eps, weight_decay
        B = self.body_params
        self.m = [torch.zeros_like(p) for p in B]
        self.v = [torch.zeros_like(p) for p in B]
        self.r = [torch.tensor(0., device=p.device) for p in B]
        self.d = [torch.tensor(eps, device=p.device) for p in B]
        self.g_prev = [torch.zeros_like(p) for p in B]
        self.t = 0

    def _train_epoch(self, lr_body, epoch):
        self.model.train()
        self.t += 1
        for batch_idx, (inp, tgt) in enumerate(tqdm(self.train_data_loader, desc='Train', ncols=100)):
            inp, tgt = inp.to(self.device), tgt.to(self.device)
            loss, acc, topk = self._forward_backward(self.model, inp, tgt)

            if self.grad_clip is not None:
                clip_grad_norm_(self.adamw_params + self.body_params, self.grad_clip)

            self.opt_adamw.step(); 
            self.opt_adamw.zero_grad()

            with torch.no_grad():
                grads = [p.grad for p in self.body_params]
                for i, (p, g) in enumerate(zip(self.body_params, grads)):
                    if g is None:
                        continue
                    p.data.mul_(1 - lr_body * self.wd)
                    scalar = torch.sum(g * torch.sign(self.g_prev[i]))
                    b2s = self.beta2 ** 0.5
                    r_new = b2s * self.r[i] + (1 - b2s) * self.d[i] * scalar
                    d_new = torch.max(self.d[i], r_new)
                    m_new = self.beta1 * self.m[i] + (1 - self.beta1) * d_new * g
                    v_new = self.beta2 * self.v[i] + (1 - self.beta2) * (d_new ** 2) * (g ** 2)
                    step = (d_new.abs() * m_new) / (torch.sqrt(v_new) + self.eps)
                    p.data.sub_(lr_body * step)
                    self.m[i], self.v[i], self.r[i], self.d[i] = m_new, v_new, r_new, d_new
                    self.g_prev[i] = g.detach().clone()

            self.train_logger.append(
                loss.item(), acc, self.grads_epochs_computed,
                time.time() - self.start_time, topk
            )
            train_step = epoch * self.train_batch_count + batch_idx
            wandb.log({
                "train/loss": loss.item(),
                "train/accuracy": acc,
                "lr_body": lr_body,
                "lr_head": self.head_lr,
                "epoch": epoch
            }, step=train_step)

class SignSGD(BaseOptimizer):
    def __init__(
        self, model, data_loaders, loss_fn, device,
        lambda_value=None, scheduler=None, scheduler_config=None,
        head_lr=0.001, head_wd=0.001, num_last_layers=2,
        grad_clip=1.0
    ):
        super().__init__(model, data_loaders, loss_fn, device,
                         lambda_value, scheduler, scheduler_config)
        self.grad_clip = grad_clip
        head_mod = self.model.head.fc
        head_params = list(head_mod.parameters())
        last_layers = self.model.layers[-num_last_layers:]
        last_params = []
        for layer in last_layers:
            last_params += [p for p in layer.parameters()]
        self.adamw_params = head_params + last_params
        adamw_ids = {id(p) for p in self.adamw_params}
        self.body_params = [p for p in self.model.parameters()
                            if p.requires_grad and id(p) not in adamw_ids]
        self.opt_adamw = torch.optim.AdamW(
            self.adamw_params,
            lr=head_lr,
            weight_decay=head_wd
        )
        self.head_lr = head_lr

    def _train_epoch(self, lr_body, epoch):
        self.model.train()
        for batch_idx, (inp, tgt) in enumerate(tqdm(self.train_data_loader, desc='Train', ncols=100)):
            inp, tgt = inp.to(self.device), tgt.to(self.device)
            loss, acc, topk = self._forward_backward(self.model, inp, tgt)

            if self.grad_clip is not None:
                clip_grad_norm_(self.adamw_params + self.body_params, self.grad_clip)

            self.opt_adamw.step(); 
            self.opt_adamw.zero_grad()

            with torch.no_grad():
                for p in self.body_params:
                    if p.grad is None:
                        continue
                    p.data.mul_(1 - lr_body * self.opt_adamw.defaults['weight_decay'])
                    p.data.sub_(lr_body * torch.sign(p.grad))

            self.train_logger.append(
                loss.item(), acc, self.grads_epochs_computed,
                time.time() - self.start_time, topk
            )
            train_step = epoch * self.train_batch_count + batch_idx
            wandb.log({
                "train/loss": loss.item(),
                "train/accuracy": acc,
                "lr_body": lr_body,
                "lr_head": self.head_lr,
                "epoch": epoch
            }, step=train_step)

class AdamW(BaseOptimizer):
    def __init__(self, model, data_loaders, loss_fn, device, lambda_value=None, 
                 scheduler=None, scheduler_config=None, 
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value, scheduler, scheduler_config)
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
        self.v = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
        self.t = 0

    def _train_epoch(self, lr, epoch):
        self.model.train()
        self.t += 1
        
        for inputs, targets in tqdm(self.train_data_loader, desc='Training', ncols=100, leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            loss, accuracy, acc_at_k = self._forward_backward(
                self.model, inputs, targets, zero_grad=True, is_test=False
            )
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed, 
                                   time.time() - self.start_time, acc_at_k)
            
            with torch.no_grad():
                for i, (param, grad) in enumerate(zip(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    [p.grad for p in self.model.parameters() if p.requires_grad]
                )):
                    self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                    self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
                    
                    m_hat = self.m[i] / (1 - self.beta1**self.t)
                    v_hat = self.v[i] / (1 - self.beta2**self.t)
                    
                    param.data.mul_(1 - lr * self.weight_decay)
                    param.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-lr)
            wandb.log({
                "train/loss": loss.item(),
                "train/accuracy": accuracy,
                "train/grad_steps": self.grads_epochs_computed,
                "epoch": epoch,
                "lr": lr
            }, step=epoch)

class AID(BaseOptimizer):
    def __init__(self, model, data_loaders, loss_fn, device, lambda_value=None, 
                 scheduler=None, scheduler_config=None, gamma=0.9):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value, scheduler, scheduler_config)
        
        self.gamma = gamma
        self.prev_grad = None
        self.prev_params = None
        self.d = 0.0
        self.lambda_sum = 0.0
        self.hat_d_sum = 0.0
        self.t = 0
        self.gamma_history = []

    def _compute_lambda(self):
        return 1.0 / (self.lambda_sum + 1e-8)**0.5 if self.t > 0 else 1.0

    def _compute_hat_d(self, current_grad):
        if self.prev_grad is None:
            return 0.0
        
        sign_prev_grad = [torch.sign(g) if g is not None else None for g in self.prev_grad]
        term = 0.0
        for g, s, gamma_val in zip(current_grad, sign_prev_grad, self.gamma_history):
            if g is not None and s is not None:
                term += torch.dot(g.flatten(), s.flatten()) * gamma_val
        self.hat_d_sum += term
        return self.hat_d_sum

    def _train_epoch(self, lr, epoch):
        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_data_loader, desc='Training', ncols=100, leave=False)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            current_params = [p.detach().clone() for p in self.model.parameters()]
            
            self.model.zero_grad()
            loss, acc, acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            current_grad = []
            for p in self.model.parameters():
                if p.requires_grad:
                    current_grad.append(p.grad.detach().clone())
                else:
                    current_grad.append(None)

            if self.prev_params is not None and self.prev_grad is not None:
                grad_diff = []
                param_diff = []
                for g, pg, p, pp in zip(current_grad, self.prev_grad, current_params, self.prev_params):
                    if g is not None and pg is not None and p.requires_grad and pp.requires_grad:
                        grad_diff.append(g - pg)
                        param_diff.append(p - pp)
                
                for gd, pd in zip(grad_diff, param_diff):
                    norm_gd = torch.norm(gd, p=1)
                    norm_pd = torch.norm(pd, p=float('inf'))
                    self.lambda_sum += (norm_gd / (norm_pd + 1e-8)).item()

            if self.t > 0:
                hat_d = self._compute_hat_d(current_grad)
                self.d = max(self.d, hat_d)

            lambda_t = self._compute_lambda()
            gamma_t = lambda_t * torch.sqrt(torch.tensor(self.d + 1e-8))

            with torch.no_grad():
                for param, grad in zip(self.model.parameters(), current_grad):
                    if param.requires_grad and grad is not None:
                        param.add_(torch.sign(grad), alpha=-lr*gamma_t.item())

            self.prev_params = current_params
            self.prev_grad = current_grad
            self.gamma_history.append(gamma_t.item())
            self.t += 1

            self.train_logger.append(
                loss.item(), 
                acc, 
                self.grads_epochs_computed, 
                time.time() - self.start_time, 
                acc_at_k
            )
            wandb.log({
                "train/loss": loss.item(),
                "train/accuracy": acc,
                "train/grad_steps": self.grads_epochs_computed,
                "epoch": epoch,
                "lr": lr
            }, step=epoch)

class Prodigy(BaseOptimizer):
    def __init__(self, model, data_loaders, loss_fn, device, lambda_value=None, 
                 scheduler=None, scheduler_config=None, 
                 beta1=0.9, beta2=0.999, d0=1e-6, eps=1e-8, weight_decay=0.01):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value, scheduler, scheduler_config)
        self.beta1 = beta1
        self.beta2 = beta2
        self.sqrt_beta2 = beta2 ** 0.5
        self.eps = eps
        self.d = [torch.full_like(p, d0) for p in self.model.parameters()]
        self.m = [torch.zeros_like(p) for p in self.model.parameters()]
        self.v = [torch.zeros_like(p) for p in self.model.parameters()]
        self.r = [torch.zeros(1, device=self.device) for _ in self.model.parameters()]
        self.s = [torch.zeros_like(p) for p in self.model.parameters()]
        self.x0 = [p.clone().detach() for p in self.model.parameters()]

    def _train_epoch(self, lr, epoch):
        self.model.train()
        for inputs, targets in tqdm(self.train_data_loader, desc='Training', ncols=100, leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            loss, accuracy, acc_at_k = self._forward_backward(
                self.model, inputs, targets, zero_grad=True, is_test=False)
            self.train_logger.append(
                loss.item(), accuracy, self.grads_epochs_computed,
                time.time() - self.start_time, acc_at_k)

            with torch.no_grad():
                for idx, param in enumerate(self.model.parameters()):
                    if not param.requires_grad:
                        continue
                    g = param.grad if param.grad is not None else torch.zeros_like(param)
                    self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * self.d[idx] * g
                    self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (self.d[idx] ** 2) * g * g
                    self.s[idx] = self.sqrt_beta2 * self.s[idx] + \
                                  (1 - self.sqrt_beta2) * lr * (self.d[idx] ** 2) * g
                    delta = (self.x0[idx] - param).detach()
                    inner = torch.sum(g * delta)
                    self.r[idx] = self.sqrt_beta2 * self.r[idx] + \
                                  (1 - self.sqrt_beta2) * lr * (self.d[idx] ** 2) * inner
                    norm_s = torch.norm(self.s[idx].view(-1), p=1)
                    d_hat = self.r[idx] / norm_s
                    self.d[idx] = torch.maximum(self.d[idx], d_hat)
                    denom = torch.sqrt(self.v[idx]) + self.d[idx] * self.eps
                    step = lr * self.d[idx] * self.m[idx] / denom
                    param.sub_(step)

            wandb.log({
                "train/loss": loss.item(),
                "train/accuracy": accuracy,
                "train/grad_steps": self.grads_epochs_computed,
                "epoch": epoch,
                "lr": lr
            }, step=epoch)

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True,
                  choices=['sgd', 'signsgd', 'adamw', 'aid', 'adamlike', 'prodigy'])

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

set_seed(152)
DEVICE = get_device(args.device)
BATCH_SIZE = 256
from models import get_swin_tiny_from_timm
MODEL = get_swin_tiny_from_timm

data_loaders = get_data_loaders(BATCH_SIZE)
EPOCHS = 20
LAMBDA_VALUE = 1e-8
from utils import get_warmup_cosine_scheduler

if args.method == 'sgd':
    opt = SGD(
        model=MODEL(DEVICE), 
        data_loaders=data_loaders, 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        device=DEVICE, 
        lambda_value=LAMBDA_VALUE,
        scheduler_config = {
            'warmup_epochs': 4,
            'base_lr': 0.05,
            'warmup_lr': 0.005 ,
            'min_lr': 0.005,
        }
    )
    opt.run(
        EPOCHS, 
        args.lr,
        scheduler_func=get_warmup_cosine_scheduler
    )
    opt.dump_json()
elif args.method == 'signsgd':
    opt = SignSGD(
        model=MODEL(DEVICE), 
        data_loaders=data_loaders, 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        device=DEVICE, 
        lambda_value=LAMBDA_VALUE,
        scheduler_config = {
            'warmup_epochs': 4,
            'base_lr': 0.05,
            'warmup_lr': 0.005 ,
            'min_lr': 0.005,
        }
    )
    opt.run(
        EPOCHS, 
        args.lr,
        scheduler_func=get_warmup_cosine_scheduler
    )
    opt.dump_json()
elif args.method == 'adamw':
    opt = AdamW(
        model=MODEL(DEVICE), 
        data_loaders=data_loaders, 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        device=DEVICE, 
        lambda_value=LAMBDA_VALUE,
        scheduler_config = {
            'warmup_epochs': 4,
            'base_lr': 0.001,
            'warmup_lr': 0.001 ,
            'min_lr': 0.001,
        }
    )
    opt.run(
        EPOCHS, 
        args.lr,
        scheduler_func=get_warmup_cosine_scheduler
    )
    opt.dump_json()
elif args.method == 'aid':
    opt = AID(
        model=MODEL(DEVICE), 
        data_loaders=data_loaders, 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        device=DEVICE, 
        lambda_value=LAMBDA_VALUE,
        scheduler_config = {
            'warmup_epochs': 4,
            'base_lr': 0.001,
            'warmup_lr': 0.001,
            'min_lr': 0.001,
        }
    )
    opt.run(
        EPOCHS, 
        args.lr,
        scheduler_func=get_warmup_cosine_scheduler
    )
    opt.dump_json()
elif args.method == 'prodigy':
    opt = Prodigy(
        model=MODEL(DEVICE), 
        data_loaders=data_loaders, 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        device=DEVICE, 
        lambda_value=LAMBDA_VALUE,
        scheduler_config = {
            'warmup_epochs': 4,
            'base_lr': 0.001,
            'warmup_lr': 0.001,
            'min_lr': 0.001,
        }
    )
    opt.run(
        EPOCHS, 
        args.lr,
        scheduler_func=get_warmup_cosine_scheduler
    )
    opt.dump_json()
elif args.method == 'adamlike':
    opt = AdamLike(
        model=MODEL(DEVICE), 
        data_loaders=data_loaders, 
        loss_fn=torch.nn.CrossEntropyLoss(), 
        device=DEVICE, 
        lambda_value=LAMBDA_VALUE,
        scheduler_config = {
            'warmup_epochs': 4,
            'base_lr': 0.001,
            'warmup_lr': 0.001 ,
            'min_lr': 0.001,
        }
    )
    opt.run(
        EPOCHS, 
        args.lr,
        scheduler_func=get_warmup_cosine_scheduler
    )
    opt.dump_json()
