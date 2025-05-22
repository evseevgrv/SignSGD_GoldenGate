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


import os
import json
import torch
from tqdm import tqdm
from utils import Logger  # Предполагается, что Logger импортируется из utils

class BaseOptimizer:
    def __init__(self, model, data_loaders, loss_fn, device, lambda_value=None):
        self.model = model
        self.train_data_loader = data_loaders[0]
        self.test_data_loader = data_loaders[1]
        self.train_batch_count = len(self.train_data_loader)
        self.test_batch_count = len(self.test_data_loader)
        self.loss_fn = loss_fn
        self.loggers = {}
        self.train_logger = Logger()
        self.test_logger = Logger()
        self.grads_epochs_computed = 0
        self.regularizer = lambda_value
        self.device = device

    def run(self, epochs, lr, exp_name=None):
        if exp_name is None:
            exp_name = f'{self.__class__.__name__}_{lr=}_{epochs=}'
            if self.regularizer is not None:
                exp_name += f'_lambda={self.regularizer}'
        for _ in tqdm(range(epochs), desc=exp_name, ncols=100):
            self._train_epoch(lr)
            tqdm.write(f'Train Loss: {self.train_logger.loss[-1]:.4f},\t Train Accuracy: {self.train_logger.accuracy[-1]:.4f}')
            self._test_epoch()
            tqdm.write(f'Test  Loss: {self.test_logger.loss[-1]:.4f},\t Test  Accuracy: {self.test_logger.accuracy[-1]:.4f}')
            tqdm.write(f'Grads epochs computed: {self.grads_epochs_computed:.2f}')
        self.loggers[exp_name] = {
            'train': self.train_logger.to_dict(),
            'test': self.test_logger.to_dict()
        }
        self.train_logger = Logger()
        self.test_logger = Logger()

    def _forward_backward(self, model, inputs, targets, zero_grad=True, is_test=False):
        if zero_grad:
            model.zero_grad()
        if is_test:
            model.eval()
        else:
            model.train()
        outputs = model(inputs)
        loss = self.loss_fn(outputs, targets)
        # Добавление L2 регуляризации
        if self.regularizer is not None:
            for param in model.parameters():
                loss += (self.regularizer/2) * torch.sum(param ** 2)

        if not is_test:
            loss.backward()
        if not is_test:
            self.grads_epochs_computed += 1 / self.train_batch_count
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        batch_accuracy = correct / targets.size(0)
        return loss, batch_accuracy

    def _test_epoch(self):
        test_loss = 0
        test_accuracy = 0
        test_data = loader_to_device(self.test_data_loader, self.device)
        for inputs, targets in tqdm(test_data, desc='Testing', ncols=100, leave=False):
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=True)
            test_loss += loss.item()
            test_accuracy += accuracy
        self.test_logger.append(test_loss / self.test_batch_count,
                                test_accuracy / self.test_batch_count,
                                self.grads_epochs_computed)

    def dump_json(self, directory='experiments'):
        os.makedirs(directory, exist_ok=True)
        for exp_name, data in self.loggers.items():
            with open(f'{directory}/{exp_name}.json', 'w') as f:
                json.dump(data, f)

    def _train_epoch(self, lr):
        """Метод, реализующий обновление параметров для одной эпохи.
        Должен быть реализован в наследниках."""
        raise NotImplementedError("Метод _train_epoch должен быть реализован в наследнике.")
    

    def _compute_full_grad(self):
        train_data = loader_to_device(self.train_data_loader, self.device)
        self.model.zero_grad()
        for inputs, targets in tqdm(train_data, desc='Computing Full Gradient', ncols=100, leave=False):
            self._forward_backward(self.model, inputs, targets, zero_grad=False, is_test=False)
        # Возвращаем усреднённый градиент по всему датасету
        return [param.grad.detach().clone() / self.train_batch_count for param in self.model.parameters()]


class SGD(BaseOptimizer):
    def _train_epoch(self, lr):
        # Перемешиваем индексы батчей
        train_data = loader_to_device(self.train_data_loader, self.device)
        for batch_num in tqdm(torch.randperm(self.train_batch_count).tolist(),
                        desc='Training', ncols=100, leave=False):
            inputs, targets = train_data[batch_num]
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed)
            # Обновление параметров по SGD
            with torch.no_grad():
                for param in self.model.parameters():
                    param.add_(param.grad, alpha=-lr)

class SVRG_Shuffle(BaseOptimizer):
    def __init__(self, model, model_ref, data_loaders, loss_fn, device, lambda_value=None, freq=3.5):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value)
        self.model_ref = model_ref
        self.freq = freq
        self.p = 1 / (freq * self.train_batch_count)
        self._g_ref = None

    def _train_epoch(self, lr):
        train_data = loader_to_device(self.train_data_loader, self.device)
        for batch_num in tqdm(torch.randperm(self.train_batch_count).tolist(),
                        desc='Training', ncols=NCOLS, leave=False):
            inputs, targets = train_data[batch_num]
            # Периодически вычисляем полный градиент
            if torch.rand(1) < self.p or self._g_ref is None:
                self._g_ref = self._compute_full_grad()
                self.model_ref.load_state_dict(self.model.state_dict())
            # Вычисляем градиенты для текущего батча на основной модели
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed)
            # Вычисляем градиенты для того же батча на эталонной модели
            _, _ = self._forward_backward(self.model_ref, inputs, targets, zero_grad=True, is_test=False)
            # Обновление параметров по схеме SVRG
            with torch.no_grad():
                for param, param_ref, grad_ref in zip(self.model.parameters(),
                                                    self.model_ref.parameters(),
                                                    self._g_ref):
                    param.add_(param.grad - param_ref.grad + grad_ref, alpha=-lr)

class SVRG_Pick(BaseOptimizer):
    def __init__(self, model, model_ref, data_loaders, loss_fn, device, lambda_value=None, freq=3.5):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value)
        self.model_ref = model_ref
        self.freq = freq
        self.m = int(self.train_batch_count/self.freq)
        self._g_ref = None

    def _train_epoch(self, lr):
        train_data = loader_to_device(self.train_data_loader, self.device)
        self._g_ref = self._compute_full_grad()
        self.model_ref.load_state_dict(self.model.state_dict())
        for batch_num, batch_idx in enumerate(tqdm(np.random.choice(self.train_batch_count, self.m).tolist(),
                        desc='Training', ncols=NCOLS, leave=False)):
            inputs, targets = train_data[batch_idx]
            # Вычисляем градиенты для текущего батча на основной модели
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed)
            # Вычисляем градиенты для того же батча на эталонной модели
            _, _ = self._forward_backward(self.model_ref, inputs, targets, zero_grad=True, is_test=False)
            # Обновление параметров по схеме SVRG
            with torch.no_grad():
                for param, param_ref, grad_ref in zip(self.model.parameters(),
                                                    self.model_ref.parameters(),
                                                    self._g_ref):
                    param.add_(param.grad - param_ref.grad + grad_ref, alpha=-lr)
    
class SARAH(BaseOptimizer):
    def __init__(self, model, model_prev, model_buffer, data_loaders, loss_fn, device, lambda_value=None, freq=1, grads_epochs_computed=0):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value)
        self.model_buffer = model_buffer
        self.model_buffer.load_state_dict(self.model.state_dict())
        self.model_prev = model_prev
        self.model_prev.load_state_dict(self.model.state_dict())
        self._g_ref = [torch.zeros_like(param) for param in self.model.parameters()] # v
        self.freq = freq
        self.m = int(self.train_batch_count/self.freq)
        self._g_ref = [torch.zeros_like(param) for param in self.model.parameters()]
        self._g_avg = [torch.zeros_like(param) for param in self.model.parameters()] # v_tilde


    def _train_epoch(self, lr):
        self._g_ref = self._compute_full_grad()
        with torch.no_grad():
            for param, grad_ref in zip(self.model.parameters(), self._g_ref):
                param.add_(grad_ref, alpha=-lr)

        train_data = loader_to_device(self.train_data_loader, self.device)
        for batch_num, batch_idx in enumerate(tqdm(np.random.choice(self.train_batch_count, self.m).tolist(),
                        desc='Training', ncols=NCOLS, leave=False)):
            inputs, targets = train_data[batch_idx]
            # Вычисляем градиенты для текущего батча на основной модели
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            g_cur = [param.grad.detach().clone() for param in self.model.parameters()]
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed)

            for idx, grad in enumerate(g_cur):
                self._g_avg[idx]  +=  grad / self.train_batch_count

            loss, accuracy = self._forward_backward(self.model_prev, inputs, targets, zero_grad=True, is_test=False)
            g_prev = [param.grad.detach().clone() for param in self.model_prev.parameters()]
            self.model_prev.load_state_dict(self.model.state_dict())
            
            for idx, _ in enumerate(self._g_ref):
                self._g_ref[idx].add_(g_cur[idx] - g_prev[idx])

            with torch.no_grad():
                for param, grad_ref in zip(self.model.parameters(), self._g_ref):
                    param.add_(grad_ref, alpha=-lr)



class NFGSVRG(BaseOptimizer):
    def __init__(self, model, model_ref, data_loaders, loss_fn, device, lambda_value=None):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value)
        self.model_ref = model_ref
        self.model_ref.load_state_dict(self.model.state_dict())
        self._g_ref = [torch.zeros_like(param) for param in self.model.parameters()] # v
        self._g_avg = [torch.zeros_like(param) for param in self.model.parameters()] # v_tilde

    def _train_epoch(self, lr):
        train_data = loader_to_device(self.train_data_loader, self.device)
        for batch_num, batch_idx in enumerate(tqdm(torch.randperm(self.train_batch_count).tolist(),
                        desc='Training', ncols=100, leave=False)):
            inputs, targets = train_data[batch_idx]
            # Вычисляем градиенты для текущего батча на основной модели
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed)
            # Вычисляем градиенты для того же батча на эталонной модели
            _, _ = self._forward_backward(self.model_ref, inputs, targets, zero_grad=True, is_test=False)
            # Обновление среднего градиента
            g_cur = [param.grad.detach().clone() for param in self.model.parameters()]
            for idx, grad in enumerate(g_cur):
                self._g_avg[idx]  +=   grad / self.train_batch_count
            # Обновление параметров по схеме SVRG
            with torch.no_grad():
                for param, param_ref, grad_ref in zip(self.model.parameters(),
                                                    self.model_ref.parameters(),
                                                    self._g_ref):
                    param.add_(param.grad - param_ref.grad + grad_ref, alpha=-lr)
        self._g_ref = [grad.clone() for grad in self._g_avg]
        for grad in self._g_avg:
            grad.zero_()
        self.model_ref.load_state_dict(self.model.state_dict())

class NFGSVRG_MID(BaseOptimizer):
    def __init__(self, model, model_ref, model_buffer, data_loaders, loss_fn, device, lambda_value=None):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value)
        self.model_ref = model_ref
        self.model_ref.load_state_dict(self.model.state_dict())
        self.model_buffer = model_buffer
        self.model_buffer.load_state_dict(self.model.state_dict())
        self._g_ref = [torch.zeros_like(param) for param in self.model.parameters()] # v
        self._g_avg = [torch.zeros_like(param) for param in self.model.parameters()] # v_tilde

    def _train_epoch(self, lr):
        train_data = loader_to_device(self.train_data_loader, self.device)
        for batch_num, batch_idx in enumerate(tqdm(torch.randperm(self.train_batch_count).tolist(),
                        desc='Training', ncols=100, leave=False)):
            inputs, targets = train_data[batch_idx]
            # Вычисляем градиенты для текущего батча на основной модели
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed)
            # Вычисляем градиенты для того же батча на эталонной модели
            _, _ = self._forward_backward(self.model_ref, inputs, targets, zero_grad=True, is_test=False)
            # Обновление среднего градиента
            g_cur = [param.grad.detach().clone() for param in self.model.parameters()]
            for idx, grad in enumerate(g_cur):
                self._g_avg[idx]  +=   grad / self.train_batch_count
            # Обновление параметров по схеме SVRG
            with torch.no_grad():
                for param, param_ref, grad_ref in zip(self.model.parameters(),
                                                    self.model_ref.parameters(),
                                                    self._g_ref):
                    param.add_(param.grad - param_ref.grad + grad_ref, alpha=-lr)
            if batch_num == self.train_batch_count//2:
                tqdm.write(f"ref update on {batch_num=}")
                self.model_buffer.load_state_dict(self.model.state_dict())
        self._g_ref = [grad.clone() for grad in self._g_avg]
        for grad in self._g_avg:
            grad.zero_()
        self.model_ref.load_state_dict(self.model_buffer.state_dict())


class NFGSVRG_AVG(BaseOptimizer):
    def __init__(self, model, model_ref, model_buffer, data_loaders, loss_fn, device, lambda_value=None):
        super().__init__(model, data_loaders, loss_fn, device, lambda_value)
        self.model_ref = model_ref
        self.model_ref.load_state_dict(self.model.state_dict())
        self.model_buffer = model_buffer
        self.model_buffer.load_state_dict(self.model.state_dict())
        self._g_ref = [torch.zeros_like(param) for param in self.model.parameters()] # v
        self._g_avg = [torch.zeros_like(param) for param in self.model.parameters()] # v_tilde

    def _train_epoch(self, lr):
        train_data = loader_to_device(self.train_data_loader, self.device)

        with torch.no_grad():
            for param in self.model_buffer.parameters():
                param.data.zero_()

        for batch_num, batch_idx in enumerate(tqdm(torch.randperm(self.train_batch_count).tolist(),
                        desc='Training', ncols=100, leave=False)):
            inputs, targets = train_data[batch_idx]
            # Вычисляем градиенты для текущего батча на основной модели
            loss, accuracy = self._forward_backward(self.model, inputs, targets, zero_grad=True, is_test=False)
            self.train_logger.append(loss.item(), accuracy, self.grads_epochs_computed)
            # Вычисляем градиенты для того же батча на эталонной модели
            _, _ = self._forward_backward(self.model_ref, inputs, targets, zero_grad=True, is_test=False)
            # Обновление среднего градиента
            g_cur = [param.grad.detach().clone() for param in self.model.parameters()]
            for idx, grad in enumerate(g_cur):
                self._g_avg[idx]  +=   grad / self.train_batch_count
            # Обновление параметров по схеме SVRG
            with torch.no_grad():
                for param, param_ref, grad_ref in zip(self.model.parameters(),
                                                    self.model_ref.parameters(),
                                                    self._g_ref):
                    param.add_(param.grad - param_ref.grad + grad_ref, alpha=-lr)
            with torch.no_grad():
            # Для каждого параметра делаем running average:
            # new_average = (batch_num/(batch_num+1))*old_average + (1/(batch_num+1))*текущее_значение
                for param_buffer, param in zip(self.model_buffer.parameters(), self.model.parameters()):
                    param_buffer.data.mul_(batch_num / (batch_num + 1))
                    param_buffer.data.add_(param.data / (batch_num + 1))
        self._g_ref = [grad.clone() for grad in self._g_avg]
        for grad in self._g_avg:
            grad.zero_()
        self.model_ref.load_state_dict(self.model_buffer.state_dict())


  



set_seed(52)
DEVICE = get_device(0)
BATCH_SIZE = 128
EPOCHS = 150
LAMBDA_VALUE = None
MODEL = get_resnet18


svrg = SVRG_Pick(
    model=MODEL(DEVICE), 
    model_ref=MODEL(DEVICE), 
    data_loaders=get_data_loaders(BATCH_SIZE),
    loss_fn=torch.nn.CrossEntropyLoss(), 
    device=DEVICE,
    lambda_value=LAMBDA_VALUE,
    freq=1
    )

nfg_svrg = NFGSVRG_AVG(
    model=MODEL(DEVICE), 
    model_ref=MODEL(DEVICE), 
    model_buffer=MODEL(DEVICE), 
    data_loaders=get_data_loaders(BATCH_SIZE),
    loss_fn=torch.nn.CrossEntropyLoss(), 
    device=DEVICE,
    lambda_value=LAMBDA_VALUE,
    )

sgd = SGD(
    model=MODEL(DEVICE), 
    data_loaders=get_data_loaders(BATCH_SIZE),
    loss_fn=torch.nn.CrossEntropyLoss(), 
    device=DEVICE,
    lambda_value=LAMBDA_VALUE,
    )


for LR in (0.05, 0.2):
    svrg.run(int(EPOCHS/3+1), LR)
    svrg.dump_json()

    nfg_svrg.run(int(EPOCHS/2+1), LR)
    nfg_svrg.dump_json()

    sgd.run(EPOCHS, LR)
    sgd.dump_json()