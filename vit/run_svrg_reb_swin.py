from utils import set_seed
from utils import get_device 
from utils import get_resnet18, get_simple_model
from utils import Logger
from utils import get_data_loaders
from utils import loader_to_device
import torch
from tqdm import tqdm
import json
import os
import numpy as np
NCOLS = 100


import time  # Для измерения времени


import os
import json
import torch
from tqdm import tqdm
from utils import Logger  # Предполагается, что Logger импортируется из utils
from utils import top_k_accuracy

import wandb
import time
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import wandb

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
        # Accumulate metrics
        total_loss = 0.0
        total_acc = 0.0
        topk_sum = {f'top@{k}': 0.0 for k in range(1, 6)}

        # Iterate over test data
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

        # Compute averages
        avg_loss = total_loss / self.test_batch_count
        avg_acc = total_acc / self.test_batch_count
        avg_topk = {k: v / self.test_batch_count for k, v in topk_sum.items()}

        # Log to internal logger
        self.test_logger.append(
            avg_loss,
            avg_acc,
            self.grads_epochs_computed,
            time.time() - self.start_time,
            avg_topk
        )

        # Log to wandb with averaged metrics at end-of-epoch step
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
        # Gradient clipping threshold
        self.grad_clip = grad_clip
        # Параметры для AdamW: head + последние N слоев
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

        # AdamW на head+последних слоях
        self.opt_adamw = torch.optim.AdamW(
            self.adamw_params, lr=head_lr, weight_decay=head_wd
        )
        self.head_lr = head_lr

        # Состояния AdamLike для body
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

            # Gradient clipping
            if self.grad_clip is not None:
                clip_grad_norm_(self.adamw_params + self.body_params, self.grad_clip)

            # AdamW на head+последних слоях
            self.opt_adamw.step(); 
            self.opt_adamw.zero_grad()

            # AdamLike на body с weight decay
            with torch.no_grad():
                grads = [p.grad for p in self.body_params]
                for i, (p, g) in enumerate(zip(self.body_params, grads)):
                    if g is None:
                        continue
                    # decoupled weight decay
                    p.data.mul_(1 - lr_body * self.wd)
                    scalar = torch.sum(g * torch.sign(self.g_prev[i]))
                    b2s = self.beta2 ** 0.5
                    r_new = b2s * self.r[i] + (1 - b2s) * self.d[i] * scalar
                    d_new = torch.max(self.d[i], r_new)
                    m_new = self.beta1 * self.m[i] + (1 - self.beta1) * d_new * g
                    v_new = self.beta2 * self.v[i] + (1 - self.beta2) * (d_new ** 2) * (g ** 2)
                    step = (d_new.abs() * m_new) / (torch.sqrt(v_new) + self.eps)
                    p.data.sub_(lr_body * step)
                    # обновляем состояния
                    self.m[i], self.v[i], self.r[i], self.d[i] = m_new, v_new, r_new, d_new
                    self.g_prev[i] = g.detach().clone()

            # Log to internal logger
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
        # Определяем слои для AdamW: head + последние N слоев из model.layers
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

            # Gradient clipping
            if self.grad_clip is not None:
                clip_grad_norm_(self.adamw_params + self.body_params, self.grad_clip)

            # Шаг AdamW для head+последних слоев
            self.opt_adamw.step(); 
            self.opt_adamw.zero_grad()

            # SignSGD для остальных с weight decay
            with torch.no_grad():
                for p in self.body_params:
                    if p.grad is None:
                        continue
                    # decoupled weight decay
                    p.data.mul_(1 - lr_body * self.opt_adamw.defaults['weight_decay'])
                    # sign update
                    p.data.sub_(lr_body * torch.sign(p.grad))

            # Log to internal logger
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


# class BaseOptimizer:
#     def __init__(self, model, data_loaders, loss_fn, device, lambda_value=None, scheduler=None, scheduler_config=None):
#         self.model = model
#         self.train_data_loader = data_loaders[0]
#         self.test_data_loader = data_loaders[1]
#         self.train_batch_count = len(self.train_data_loader)
#         self.test_batch_count = len(self.test_data_loader)
#         self.loss_fn = loss_fn
#         self.loggers = {}
#         self.train_logger = Logger()
#         self.test_logger = Logger()
#         self.grads_epochs_computed = 0
#         self.regularizer = lambda_value
#         self.device = device
#         self.scheduler = scheduler
#         self.scheduler_config = scheduler_config or {}

#         # Разделяем параметры на голову и тело
#         self.head_params = list(model.head.fc.parameters())
#         all_params = list(model.parameters())
#         self.body_params = [p for p in all_params if p not in set(self.head_params)]
        
#         # Инициализируем оптимизатор для головы
#         self.head_optimizer = torch.optim.AdamW(
#             self.head_params,
#             lr=1e-3,  # Временное значение, будет обновлено
#             weight_decay=lambda_value if lambda_value is not None else 0.0
#         )

#     def run(self, epochs, lr, exp_name=None, scheduler_func=None):
#         self.start_time = time.time()  # Засекаем время начала всего обучения
#         if exp_name is None:
#             exp_name = f'{self.__class__.__name__}_{lr=}_{epochs=}'
#             if self.regularizer is not None:
#                 exp_name += f'_lambda={self.regularizer}'
#         config = {
#             "method": self.__class__.__name__,
#             "lr": lr,
#             "epochs": epochs,
#             "lambda": self.regularizer,
#             "batch_size": BATCH_SIZE,
#             "model": "swin_tiny"
#         }
        
#         if exp_name is None:
#             exp_name = f'{self.__class__.__name__}_{lr=}_{epochs=}'
#             if self.regularizer is not None:
#                 exp_name += f'_lambda={self.regularizer}'
        
#         wandb.init(
#             project="vit-experiments",
#             name=exp_name,
#             config=config,
#             tags=[args.method, "tiny-imagenet"]
#         )

#         if scheduler_func is not None:
#             config = {
#                 'warmup_epochs': 5,
#                 'total_epochs': epochs,
#                 'base_lr': lr,
#                 'warmup_lr': lr * 0.1,
#                 'min_lr': 1e-5,
#                 **self.scheduler_config 
#             }
#             self.scheduler = scheduler_func(**config)
#         else:
#             self.scheduler = lambda epoch: lr 

#         for epoch in tqdm(range(epochs), desc=exp_name, ncols=100):
#             current_lr = self.scheduler(epoch)
#             self._train_epoch(current_lr, epoch)
#             tqdm.write(f'Train Loss: {self.train_logger.loss[-1]:.4f},\t Train Accuracy: {self.train_logger.accuracy[-1]:.4f}')
#             self._test_epoch(epoch)
#             tqdm.write(f'Test  Loss: {self.test_logger.loss[-1]:.4f},\t Test  Accuracy: {self.test_logger.accuracy[-1]:.4f}')
#             tqdm.write(f'Grads epochs computed: {self.grads_epochs_computed:.2f}')
#             total_time = time.time() - self.start_time
#             tqdm.write(f'Training Time: {total_time:.2f}s')

#         self.loggers[exp_name] = {
#             'train': self.train_logger.to_dict(),
#             'test': self.test_logger.to_dict(),
#         }
#         self.train_logger = Logger()
#         self.test_logger = Logger()

#         wandb.finish()


#     def _forward_backward(self, model, inputs, targets, zero_grad=True, is_test=False):
#         if zero_grad:
#             model.zero_grad()
#         if is_test:
#             model.eval()
#         else:
#             model.train()
#         outputs = model(inputs)
#         bs, num_logits = outputs.shape
#         tmin, tmax = targets.min().item(), targets.max().item()
#         if tmin < 0 or tmax >= num_logits:
#             raise RuntimeError(
#                 f"Неправильные метки для CrossEntropy: targets в диапазоне [{tmin}, {tmax}], "
#                 f"а outputs.shape = {outputs.shape} (ожидалось num_logits > {tmax})"
#             )
#         loss = self.loss_fn(outputs, targets)

#         # Добавляем L2 регуляризацию только для тела
#         if self.regularizer is not None:
#             for param in self.body_params:
#                 if param.requires_grad:
#                     loss += (self.regularizer/2) * torch.sum(param ** 2)

#         if not is_test:
#             loss.backward()
#         if not is_test:
#             self.grads_epochs_computed += 1 / self.train_batch_count
#         _, predicted = outputs.max(1)
#         correct = predicted.eq(targets).sum().item()
#         batch_accuracy = correct / targets.size(0)

#         acc_at_k = top_k_accuracy(outputs, targets, ks=(1, 2, 3, 4, 5))
#         return loss, batch_accuracy, acc_at_k

#     def _test_epoch(self, epoch):
#         test_loss = 0
#         test_accuracy = 0
#         test_data = loader_to_device(self.test_data_loader, self.device)
#         topk_aggregate = {f'top@{k}': 0.0 for k in range(1, 6)}
#         for inputs, targets in tqdm(test_data, desc='Testing', ncols=100, leave=False):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             loss, acc, acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=True)
#             test_loss += loss.item()
#             test_accuracy += acc
#             for k in acc_at_k:
#                 topk_aggregate[k] += acc_at_k[k]
#         avg_topk = {k: v / self.test_batch_count for k, v in topk_aggregate.items()}

#         self.test_logger.append(test_loss / self.test_batch_count,
#                                 test_accuracy / self.test_batch_count,
#                                 self.grads_epochs_computed,
#                                 time.time() - self.start_time,
#                                 avg_topk)
#         wandb.log({
#             "test/loss": test_loss / self.test_batch_count,
#             "test/accuracy": test_accuracy / self.test_batch_count,
#             "test/top1_acc": avg_topk['top@1'],
#             "test/top5_acc": avg_topk['top@5'],
#             "epoch": epoch
#         }, step=epoch)


#     def dump_json(self, directory='experiments'):
#         os.makedirs(directory, exist_ok=True)
#         for exp_name, data in self.loggers.items():
#             with open(f'{directory}/{exp_name}.json', 'w') as f:
#                 json.dump(data, f)

#     def _train_epoch(self, lr):
#         """Метод, реализующий обновление параметров для одной эпохи.
#         Должен быть реализован в наследниках."""
#         raise NotImplementedError("Метод _train_epoch должен быть реализован в наследнике.")
    

#     def _compute_full_grad(self):
#         self.model.zero_grad()
#         for inputs, targets in tqdm(self.train_data_loader, desc='Training', ncols=100, leave=False):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             self._forward_backward(self.model, inputs, targets, zero_grad=False, is_test=False)
#         # Возвращаем усреднённый градиент по всему датасету
#         return [param.grad.detach().clone() / self.train_batch_count for param in self.model.parameters() if param.requires_grad]


# class SGD(BaseOptimizer):
#     def _train_epoch(self, lr, epoch):
#         self.model.train()
#         for inputs, targets in tqdm(self.train_data_loader, desc='Training', ncols=100, leave=False):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             loss, accuracy, acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
#             self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed, time.time() - self.start_time, acc_at_k)
#             # Обновление параметров по SGD
#             with torch.no_grad():
#                 for param in self.model.parameters():
#                     if param.requires_grad:
#                         param.add_(param.grad, alpha=-lr)
#             wandb.log({
#                 "train/loss": loss.item(),
#                 "train/accuracy": accuracy,
#                 "train/grad_steps": self.grads_epochs_computed,
#                 "epoch": epoch,
#                 "lr": lr
#             }, step=epoch)


# class BaseOptimizer:
#     def __init__(
#         self, model, data_loaders, loss_fn, device,
#         lambda_value=None, scheduler=None, scheduler_config=None
#     ):
#         self.model = model
#         self.train_data_loader = data_loaders[0]
#         self.test_data_loader = data_loaders[1]
#         self.train_batch_count = len(self.train_data_loader)
#         self.test_batch_count = len(self.test_data_loader)
#         self.loss_fn = loss_fn
#         self.train_logger = Logger()
#         self.test_logger = Logger()
#         self.grads_epochs_computed = 0
#         self.regularizer = lambda_value
#         self.device = device
#         self.scheduler = scheduler
#         self.scheduler_config = scheduler_config or {}
#         self.loggers = {}

#     def run(self, epochs, lr, exp_name=None, scheduler_func=None):
#         if exp_name is None:
#             exp_name = f"{self.__class__.__name__}_lr={lr}_epochs={epochs}"
#             if self.regularizer is not None:
#                 exp_name += f"_lambda={self.regularizer}"
#         config = {
#             "method": self.__class__.__name__,
#             "lr": lr,
#             "epochs": epochs,
#             "lambda": self.regularizer,
#             "batch_size": BATCH_SIZE,
#             "model": "swin_tiny"
#         }
#         wandb.init(
#             project="vit-experiments",
#             name=exp_name,
#             config=config,
#             tags=[args.method, "tiny-imagenet"]
#         )
#         if scheduler_func is not None:
#             sched_cfg = {
#                 'warmup_epochs': 5,
#                 'total_epochs': epochs,
#                 'base_lr': lr,
#                 'warmup_lr': lr * 0.1,
#                 'min_lr': 1e-5,
#                 **self.scheduler_config
#             }
#             self.scheduler = scheduler_func(**sched_cfg)
#         else:
#             self.scheduler = lambda e: lr

#         self.start_time = time.time()
#         for epoch in tqdm(range(epochs), desc=exp_name, ncols=100):
#             curr_lr = self.scheduler(epoch)
#             self._train_epoch(curr_lr, epoch)
#             self._test_epoch(epoch)
            
#             tqdm.write(
#                 f"Train Loss: {self.train_logger.loss[-1]:.4f}, "
#                 f"Train Acc: {self.train_logger.accuracy[-1]:.4f}"
#             )
#             tqdm.write(
#                 f"Test Loss: {self.test_logger.loss[-1]:.4f}, "
#                 f"Test Acc: {self.test_logger.accuracy[-1]:.4f}"
#             )
#         self.loggers[exp_name] = {
#             'train': self.train_logger.to_dict(),
#             'test': self.test_logger.to_dict(),
#         }
#         wandb.finish()

#     def _forward_backward(self, model, inputs, targets, zero_grad=True, is_test=False):
#         if zero_grad:
#             model.zero_grad()
#         model.eval() if is_test else model.train()
#         outputs = model(inputs)
#         bs, num_logits = outputs.shape
#         tmin, tmax = targets.min().item(), targets.max().item()
#         if tmin < 0 or tmax >= num_logits:
#             raise RuntimeError(
#                 f"Неправильные метки: targets в [{tmin},{tmax}], outputs.shape={outputs.shape}"
#             )
#         loss = self.loss_fn(outputs, targets)
#         if self.regularizer is not None:
#             for p in model.parameters():
#                 loss += (self.regularizer / 2) * torch.sum(p ** 2)
#         if not is_test:
#             loss.backward()
#             self.grads_epochs_computed += 1 / self.train_batch_count
#         _, predicted = outputs.max(1)
#         correct = predicted.eq(targets).sum().item()
#         acc = correct / targets.size(0)
#         topk = top_k_accuracy(outputs, targets, ks=(1, 2, 3, 4, 5))
#         return loss, acc, topk

#     def _test_epoch(self, epoch):
#         # Accumulate metrics
#         total_loss = 0.0
#         total_acc = 0.0
#         topk_sum = {f'top@{k}': 0.0 for k in range(1, 6)}

#         # Iterate over test data
#         for inputs, targets in tqdm(
#             loader_to_device(self.test_data_loader, self.device),
#             desc='Testing', ncols=100, leave=False
#         ):
#             loss, acc, topk = self._forward_backward(
#                 self.model,
#                 inputs.to(self.device),
#                 targets.to(self.device),
#                 zero_grad=True,
#                 is_test=True
#             )
#             total_loss += loss.item()
#             total_acc += acc
#             for k, v in topk.items():
#                 topk_sum[k] += v

#         # Compute averages
#         avg_loss = total_loss / self.test_batch_count
#         avg_acc = total_acc / self.test_batch_count
#         avg_topk = {k: v / self.test_batch_count for k, v in topk_sum.items()}

#         # Log to internal logger
#         self.test_logger.append(
#             avg_loss,
#             avg_acc,
#             self.grads_epochs_computed,
#             time.time() - self.start_time,
#             avg_topk
#         )

#         # Log to wandb with averaged metrics
#         wandb.log({
#             "test/loss": avg_loss,
#             "test/acc": avg_acc,
#             "test/top1_acc": avg_topk['top@1'],
#             "test/top5_acc": avg_topk['top@5'],
#             "epoch": epoch
#         }, step=epoch)

#     def dump_json(self, directory='experiments'):
#         os.makedirs(directory, exist_ok=True)
#         for name, logs in self.loggers.items():
#             with open(f'{directory}/{name}.json', 'w') as f:
#                 json.dump(logs, f)

# class SGD(BaseOptimizer):
#     def __init__(
#         self, model, data_loaders, loss_fn, device,
#         lambda_value=None, scheduler=None, scheduler_config=None,
#         head_lr=None, head_wd=0.0
#     ):
#         super().__init__(model, data_loaders, loss_fn, device,
#                          lambda_value, scheduler, scheduler_config)
#         # Разделяем параметры головы и тела
#         head_module = self.model.head.fc
#         self.head_params = list(head_module.parameters())
#         head_ids = {id(p) for p in self.head_params}
#         self.body_params = [p for p in self.model.parameters()
#                             if p.requires_grad and id(p) not in head_ids]
#         # Оптимизатор AdamW для головы
#         self.head_lr = 0.001
#         self.opt_head = torch.optim.AdamW(
#             self.head_params, lr=self.head_lr, weight_decay=0
#         )

#     def _train_epoch(self, lr_body, epoch):
#         self.model.train()
#         for inputs, targets in tqdm(self.train_data_loader,
#                                      desc='Training', ncols=100, leave=False):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             loss, acc, topk = self._forward_backward(
#                 self.model, inputs, targets, zero_grad=True, is_test=False
#             )
#             self.train_logger.append(
#                 loss.item(), acc, self.grads_epochs_computed,
#                 time.time() - self.start_time, topk
#             )
#             wandb.log({
#                 "train/loss": loss.item(),
#                 "train/accuracy": acc,
#                 "train/grad_steps": self.grads_epochs_computed,
#                 "epoch": epoch,
#                 "lr_body": lr_body,
#                 "lr_head": self.head_lr
#             }, step=epoch)
#             # Шаг AdamW по голове
#             self.opt_head.step()
#             self.opt_head.zero_grad()
#             # Ручной SGD-шаг по остальным параметрам
#             with torch.no_grad():
#                 for p in self.body_params:
#                     if p.grad is not None:
#                         p.add_(p.grad, alpha=-lr_body)


# class SignSGD(BaseOptimizer):
#     def _train_epoch(self, lr, epoch):
#         self.model.train()
#         # for param_group in self.head_optimizer.param_groups:
#         #     param_group['lr'] = lr
            
#         for inputs, targets in tqdm(self.train_data_loader, desc='Training', ncols=100, leave=False):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             loss, accuracy, acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
#             self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed, time.time() - self.start_time, acc_at_k)

#             # Обновляем тело через SignSGD
#             with torch.no_grad():
#                 for param in self.body_params:
#                     if param.requires_grad:
#                         param.add_(torch.sign(param.grad), alpha=-lr)
            
#             self.head_optimizer.step()
#             self.head_optimizer.zero_grad()
            
#             for param in self.body_params:
#                 if param.grad is not None:
#                     param.grad.zero_()
            
#             wandb.log({
#                 "train/loss": loss.item(),
#                 "train/accuracy": accuracy,
#                 "train/grad_steps": self.grads_epochs_computed,
#                 "epoch": epoch,
#                 "lr": lr
#             }, step=epoch)

# class SignSGD(BaseOptimizer):
#     def __init__(
#         self, model, data_loaders, loss_fn, device,
#         lambda_value=None, scheduler=None, scheduler_config=None,
#         head_lr=None, head_wd=0.0
#     ):
#         super().__init__(model, data_loaders, loss_fn, device,
#                          lambda_value, scheduler, scheduler_config)
#         # Разделяем параметры головы и тела
#         head_module = self.model.head.fc
#         self.head_params = list(head_module.parameters())
#         head_ids = {id(p) for p in self.head_params}
#         self.body_params = [p for p in self.model.parameters()
#                             if p.requires_grad and id(p) not in head_ids]
#         # Оптимизатор AdamW для головы
#         self.head_lr = 0.001
#         self.opt_head = torch.optim.AdamW(
            # self.head_params, lr=0.001, weight_decay=0.001
    #     )

    # def _train_epoch(self, lr_body, epoch):
    #     self.model.train()
    #     for inputs, targets in tqdm(self.train_data_loader,
    #                                  desc='Training', ncols=100, leave=False):
    #         inputs, targets = inputs.to(self.device), targets.to(self.device)
    #         loss, acc, topk = self._forward_backward(
    #             self.model, inputs, targets, zero_grad=True, is_test=False
    #         )
    #         self.train_logger.append(
    #             loss.item(), acc, self.grads_epochs_computed,
    #             time.time() - self.start_time, topk
    #         )
    #         wandb.log({
    #             "train/loss": loss.item(),
    #             "train/accuracy": acc,
    #             "train/grad_steps": self.grads_epochs_computed,
    #             "epoch": epoch,
    #             "lr_body": lr_body,
    #             "lr_head": self.head_lr
    #         }, step=epoch)
    #         # Шаг AdamW по голове
    #         self.opt_head.step()
    #         self.opt_head.zero_grad()
    #         # Ручной SGD-шаг по остальным параметрам
    #         with torch.no_grad():
    #             for p in self.body_params:
    #                 if p.grad is not None:
    #                     p.add_(p.grad.sign(), alpha=-lr_body)

import argparse
import os
import time
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from timm import create_model
import wandb

from utils import (
    set_seed, get_device, get_data_loaders,
    loader_to_device, Logger, top_k_accuracy,
    get_warmup_cosine_scheduler
)

# Функция для загрузки модели Swin Tiny

# def get_swin_tiny_from_timm(device, num_classes=1000):
#     model = create_model(
#         'swin_tiny_patch4_window7_224.ms_in22k',
#         pretrained=True,
#     )
#     # Заморозка параметров
#     for p in model.parameters(): p.requires_grad = False
#     total_stages = len(model.layers)
#     for i, stage in enumerate(model.layers):
#         if i >= total_stages // 2:
#             for p in stage.parameters(): p.requires_grad = True
#     for p in model.norm.parameters(): p.requires_grad = True
#     # Замена головы
#     in_f = model.head.fc.in_features
#     model.head.fc = nn.Linear(in_f, num_classes)
#     for p in model.head.fc.parameters(): p.requires_grad = True
#     return model.to(device)

# class BaseOptimizer:
#     def __init__(self, model, data_loaders, loss_fn, device,
#                  lambda_value=None, scheduler=None, scheduler_config=None):
#         self.model = model
#         self.train_data_loader, self.test_data_loader = data_loaders
#         self.train_batch_count = len(self.train_data_loader)
#         self.test_batch_count = len(self.test_data_loader)
#         self.loss_fn = loss_fn
#         self.device = device
#         self.regularizer = lambda_value
#         self.scheduler = scheduler
#         self.scheduler_config = scheduler_config or {}
#         self.train_logger = Logger()
#         self.test_logger = Logger()
#         self.grads_epochs_computed = 0
#         self.loggers = {}

#     def run(self, epochs, lr, exp_name=None, scheduler_func=None):
#         if exp_name is None:
#             exp_name = f"{self.__class__.__name__}_lr={lr}_epo={epochs}"
#         config = {"method": self.__class__.__name__, "lr": lr,
#                   "epochs": epochs, "lambda": self.regularizer}
#         wandb.init(project="vit-experiments", name=exp_name,
#                    config=config)
#         if scheduler_func:
#             cfg = {'warmup_epochs':5,'total_epochs':epochs,
#                    'base_lr':lr,'warmup_lr':lr*0.1,'min_lr':1e-5}
#             cfg.update(self.scheduler_config)
#             self.scheduler = scheduler_func(**cfg)
#         else:
#             self.scheduler = lambda e: lr
#         self.start_time = time.time()
#         for epoch in range(epochs):
#             curr_lr = self.scheduler(epoch)
#             self._train_epoch(curr_lr, epoch)
#             self._test_epoch(epoch)
#         wandb.finish()

#     def _forward_backward(self, model, inputs, targets, zero_grad=True, is_test=False):
#         if zero_grad: model.zero_grad()
#         model.eval() if is_test else model.train()
#         outputs = model(inputs)
#         bs, num_logits = outputs.shape
#         tmin, tmax = targets.min().item(), targets.max().item()
#         if tmin<0 or tmax>=num_logits:
#             raise RuntimeError(f"Labels {tmin}-{tmax} mismatch logits {num_logits}")
#         loss = self.loss_fn(outputs, targets)
#         if self.regularizer:
#             for p in model.parameters(): loss += (self.regularizer/2)*torch.sum(p**2)
#         if not is_test:
#             loss.backward()
#             self.grads_epochs_computed += 1/self.train_batch_count
#         pred = outputs.argmax(1)
#         acc = pred.eq(targets).float().mean().item()
#         topk = top_k_accuracy(outputs, targets, ks=(1,2,3,4,5))
#         return loss, acc, topk

#     def _test_epoch(self, epoch):
#         total_loss=0; total_acc=0
#         topk_sum={f'top@{k}':0 for k in range(1,6)}
#         for inputs,targets in loader_to_device(self.test_data_loader,self.device):
#             loss, acc, topk = self._forward_backward(
#                 self.model, inputs, targets, zero_grad=True, is_test=True)
#             total_loss+=loss.item(); total_acc+=acc
#             for k,v in topk.items(): topk_sum[k]+=v
#         avg_topk={k:v/self.test_batch_count for k,v in topk_sum.items()}
#         wandb.log({"test/loss":total_loss/self.test_batch_count,
#                    "test/accuracy":total_acc/self.test_batch_count,
#                    "test/top1":avg_topk['top@1'],"epoch":epoch})


# run_svrg_reb_swin.py

class AdamW(BaseOptimizer):
    def __init__(self, model, data_loaders, loss_fn, device, lambda_value=None, 
                 scheduler=None, scheduler_config=None, 
                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value, scheduler, scheduler_config)
        
        # Параметры AdamW
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Инициализация моментов
        self.m = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
        self.v = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
        self.t = 0

    def _train_epoch(self, lr, epoch):
        self.model.train()
        self.t += 1
        
        for inputs, targets in tqdm(self.train_data_loader, desc='Training', ncols=100, leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward + backward
            loss, accuracy, acc_at_k = self._forward_backward(
                self.model, inputs, targets, zero_grad=True, is_test=False
            )
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed, 
                                   time.time() - self.start_time, acc_at_k)
            
            # Обновление параметров
            with torch.no_grad():
                for i, (param, grad) in enumerate(zip(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    [p.grad for p in self.model.parameters() if p.requires_grad]
                )):
                    # Обновление моментов
                    self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                    self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)
                    
                    # Bias correction
                    m_hat = self.m[i] / (1 - self.beta1**self.t)
                    v_hat = self.v[i] / (1 - self.beta2**self.t)
                    
                    # Weight decay и обновление
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
        self.prev_grad = None  # Теперь содержит None для необучаемых параметров
        self.prev_params = None  # Содержит все параметры
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
            # Сохраняем текущие параметры (все)
            current_params = [p.detach().clone() for p in self.model.parameters()]
            
            # if self.prev_params is not None:
                # Восстанавливаем prev_params (все параметры)
                # with torch.no_grad():
                #     for p, prev_p in zip(self.model.parameters(), self.prev_params):
                #         p.copy_(prev_p)
                
                # # Вычисляем градиент для prev_params
                # self.model.zero_grad()
                # _, _, _ = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
                # # Сохраняем градиенты (включая None)

                # prev_grad = []
                # for p in self.model.parameters():
                #     if p.requires_grad:
                #         prev_grad.append(p.grad.detach().clone())
                #     else:
                #         prev_grad.append(None)
                # self.prev_grad = prev_grad
                
            #     # Восстанавливаем текущие параметры
            #     with torch.no_grad():
            #         for p, curr_p in zip(self.model.parameters(), current_params):
            #             p.copy_(curr_p)
            # else:
            #     self.prev_grad = None

            # Основной forward-backward
            self.model.zero_grad()
            loss, acc, acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            # Собираем градиенты для всех параметров
            current_grad = []
            for p in self.model.parameters():
                if p.requires_grad:
                    current_grad.append(p.grad.detach().clone())
                else:
                    current_grad.append(None)

            # Обновление lambda_sum
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

            # Обновление hat_d и d
            if self.t > 0:
                hat_d = self._compute_hat_d(current_grad)
                self.d = max(self.d, hat_d)

            # Вычисление gamma_t
            lambda_t = self._compute_lambda()
            gamma_t = lambda_t * torch.sqrt(torch.tensor(self.d + 1e-8))

            # Обновление параметров
            with torch.no_grad():
                for param, grad in zip(self.model.parameters(), current_grad):
                    if param.requires_grad and grad is not None:
                        param.add_(torch.sign(grad), alpha=-lr*gamma_t.item())

            # Сохраняем состояние
            self.prev_params = current_params
            self.prev_grad = current_grad
            self.gamma_history.append(gamma_t.item())
            self.t += 1

            # Логирование
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


# class AdamLike(BaseOptimizer):
#     def __init__(
#         self, model, data_loaders, loss_fn, device, lambda_value=None,
#         scheduler=None, scheduler_config=None,
#         beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
#         head_lr=0.0, head_wd=0.0
#     ):
#         super().__init__(model, data_loaders, loss_fn, device,
#                          lambda_value, scheduler, scheduler_config)
#         # Разделяем параметры головы и тела
#         head_module = self.model.head.fc
#         self.head_params = list(head_module.parameters())
#         head_ids = {id(p) for p in self.head_params}
#         self.body_params = [p for p in self.model.parameters()
#                             if p.requires_grad and id(p) not in head_ids]
#         # Инициализация состояний AdamLike для body
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = eps
#         self.weight_decay = weight_decay
#         params_body = self.body_params
#         self.m = [torch.zeros_like(p) for p in params_body]
#         self.v = [torch.zeros_like(p) for p in params_body]
#         self.r = [torch.tensor(0.0, device=p.device) for p in params_body]
#         self.d = [torch.tensor(eps, device=p.device) for p in params_body]
#         self.g_prev = [torch.zeros_like(p) for p in params_body]
#         self.t = 0
#         # Оптимизатор AdamW для головы
#         self.opt_head = torch.optim.AdamW(
#             self.head_params, lr=0.001, weight_decay=0.001
#         )
#         self.head_lr = head_lr

#     def _train_epoch(self, lr_body, epoch):
#         self.model.train()
#         self.t += 1
#         for inputs, targets in tqdm(self.train_data_loader,
#                                      desc='Training', ncols=100, leave=False):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             loss, accuracy, acc_at_k = self._forward_backward(
#                 self.model, inputs, targets, zero_grad=True, is_test=False
#             )
#             self.train_logger.append(
#                 loss.item(), accuracy, self.grads_epochs_computed,
#                 time.time() - self.start_time, acc_at_k
#             )
#             # Шаг AdamW по голове
#             self.opt_head.step()
#             self.opt_head.zero_grad()
#             # Шаг AdamLike по body
#             with torch.no_grad():
#                 grads = [p.grad for p in self.body_params]
#                 for i, (param, grad) in enumerate(zip(self.body_params, grads)):
#                     g_prev = self.g_prev[i]
#                     r_prev = self.r[i]
#                     d_prev = self.d[i]
#                     scalar = torch.sum(grad * torch.sign(g_prev))
#                     sqrt_b2 = self.beta2 ** 0.5
#                     r_new = sqrt_b2 * r_prev + (1 - sqrt_b2) * d_prev * scalar
#                     d_new = torch.max(d_prev, r_new)
#                     m_new = self.beta1 * self.m[i] + (1 - self.beta1) * d_new * grad
#                     v_new = self.beta2 * self.v[i] + (1 - self.beta2) * (d_new ** 2) * (grad ** 2)
#                     step = (d_new.abs() * m_new) / (torch.sqrt(v_new) + self.eps)
#                     param.data.sub_(lr_body * step)
#                     # Обновляем состояния
#                     self.m[i], self.v[i], self.r[i], self.d[i] = m_new, v_new, r_new, d_new
#                     self.g_prev[i] = grad.detach().clone()
#             # Логирование
#             wandb.log({
#                 "train/loss": loss.item(),
#                 "train/accuracy": accuracy,
#                 "train/grad_steps": self.grads_epochs_computed,
#                 "epoch": epoch,
#                 "lr_body": lr_body,
#                 "lr_head": self.head_lr
#             }, step=epoch)

import math

import time
import torch
from tqdm import tqdm
import wandb

class Prodigy(BaseOptimizer):
    def __init__(self, model, data_loaders, loss_fn, device, lambda_value=None, 
                 scheduler=None, scheduler_config=None, 
                 beta1=0.9, beta2=0.999, d0=1e-6, eps=1e-8, weight_decay=0.01):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value, scheduler, scheduler_config)
        self.beta1 = beta1
        self.beta2 = beta2
        self.sqrt_beta2 = beta2 ** 0.5
        self.eps = eps
        # Скаляры d для каждого параметра
        self.d = [torch.full_like(p, d0) for p in self.model.parameters()]
        # Инициализация моментов и накопителей для каждого параметра
        self.m = [torch.zeros_like(p) for p in self.model.parameters()]
        self.v = [torch.zeros_like(p) for p in self.model.parameters()]
        self.r = [torch.zeros(1, device=self.device) for _ in self.model.parameters()]
        self.s = [torch.zeros_like(p) for p in self.model.parameters()]
        # Сохранение начального состояния параметров
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
                # Для каждого параметра отдельно
                for idx, param in enumerate(self.model.parameters()):
                    if not param.requires_grad:
                        continue
                    # Градиент
                    g = param.grad if param.grad is not None else torch.zeros_like(param)
                    # m_{k+1}
                    self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * self.d[idx] * g
                    # v_{k+1}
                    self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (self.d[idx] ** 2) * g * g
                    # s_{k+1}
                    self.s[idx] = self.sqrt_beta2 * self.s[idx] + \
                                  (1 - self.sqrt_beta2) * lr * (self.d[idx] ** 2) * g
                    # ⊗ (p0 - p)
                    delta = (self.x0[idx] - param).detach()
                    # скалярное произведение <g, x0-x>
                    inner = torch.sum(g * delta)
                    # r_{k+1}
                    self.r[idx] = self.sqrt_beta2 * self.r[idx] + \
                                  (1 - self.sqrt_beta2) * lr * (self.d[idx] ** 2) * inner
                    # d_hat_{k+1}
                    norm_s = torch.norm(self.s[idx].view(-1), p=1)
                    d_hat = self.r[idx] / norm_s
                    # d_{k+1}
                    self.d[idx] = torch.maximum(self.d[idx], d_hat)
                    # обновление параметра x_{k+1}
                    denom = torch.sqrt(self.v[idx]) + self.d[idx] * self.eps
                    step = lr * self.d[idx] * self.m[idx] / denom
                    param.sub_(step)

            wandb.log({
                "train/loss": loss.item(),
                "train/accuracy": accuracy,
                "train/grad_steps": self.grads_epochs_computed,
                "epoch": epoch,
                "lr": lr
                # "prodigy/d": self.d,
                # "prodigy/r": self.r.item()
            }, step=epoch)

        # return self.model


# class Prodigy(BaseOptimizer):
#     def __init__(self, model, data_loaders, loss_fn, device, lambda_value=None, 
#                  scheduler=None, scheduler_config=None, 
#                  beta1=0.9, beta2=0.999, d0=1e-6, eps=1e-8, weight_decay=0.01):
#         super().__init__(model, data_loaders, loss_fn, device, lambda_value, scheduler, scheduler_config)
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.d = d0
#         self.eps = eps
#         self.r = 0.0
#         self.params = [p for p in model.parameters() if p.requires_grad]
        
#         # Инициализация состояний m, v, s
#         self.m = [torch.zeros_like(p) for p in self.params]
#         self.v = [torch.zeros_like(p) for p in self.params]
#         self.s = [torch.zeros_like(p) for p in self.params]
        
#         # Сохраняем начальные параметры x0 при первом вызове _train_epoch
#         self.x0 = None

#     def _train_epoch(self, lr, epoch):
#         self.model.train()
#         if self.x0 is None:
#             self.x0 = [p.clone().detach() for p in self.params]
        
#         for inputs, targets in tqdm(self.train_data_loader, desc='Training', ncols=100, leave=False):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             loss, accuracy, acc_at_k = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
#             self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed, time.time() - self.start_time, acc_at_k)
            
#             with torch.no_grad():
#                 grads = [p.grad for p in self.params]
#                 beta1, beta2 = self.beta1, self.beta2
#                 sqrt_beta2 = math.sqrt(beta2)
#                 gamma_k = lr
#                 d_k = self.d
                
#                 # Обновляем m, v, s для каждого параметра
#                 new_m, new_v, new_s = [], [], []
#                 for i, (p, g) in enumerate(zip(self.params, grads)):
#                     # Обновление моментов
#                     m_new = beta1 * self.m[i] + (1 - beta1) * d_k * g
#                     v_new = beta2 * self.v[i] + (1 - beta2) * (d_k ** 2) * (g ** 2)
#                     s_new = sqrt_beta2 * self.s[i] + (1 - sqrt_beta2) * gamma_k * (d_k ** 2) * g
#                     new_m.append(m_new)
#                     new_v.append(v_new)
#                     new_s.append(s_new)
                
#                 # Обновление r
#                 sum_g_x0x = sum(torch.sum(g * (x0 - p)) for g, p, x0 in zip(grads, self.params, self.x0))
#                 self.r = sqrt_beta2 * self.r + (1 - sqrt_beta2) * gamma_k * (d_k ** 2) * sum_g_x0x
                
#                 # Вычисление d_hat
#                 total_s_l1 = sum(torch.sum(torch.abs(s)) for s in new_s)
#                 d_hat = self.r / (total_s_l1 + self.eps)
#                 self.d = max(d_k, d_hat.item())
                
#                 # Обновление параметров
#                 for i, p in enumerate(self.params):
#                     denominator = torch.sqrt(new_v[i] + d_k * self.eps)
#                     update = gamma_k * d_k * new_m[i] / denominator
#                     p.sub_(update)
                
#                 # Сохраняем новые состояния
#                 self.m, self.v, self.s = new_m, new_v, new_s
                
#                 # Логирование
#                 wandb.log({
#                     "train/loss": loss.item(),
#                     "train/accuracy": accuracy,
#                     "train/grad_steps": self.grads_epochs_computed,
#                     "epoch": epoch,
#                     "lr": lr,
#                     "prodigy/d": self.d,
#                     "prodigy/r": self.r.item()
#                 }, step=epoch)


import argparse

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
    # opt=SignSGD(model_fn(DEVICE),data_loaders,
    #                   nn.CrossEntropyLoss(),DEVICE,
    #                   lambda_value=LAMBDA,scheduler_config={},
    #                   head_lr=args.head_lr,head_wd=1e-2,
    #                   num_last_layers=args.num_last_layers)
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
