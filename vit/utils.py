import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _load_cifar(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

def _load_imagenet(batch_size=128):
    import os
    from torchvision.datasets import ImageFolder
    from torchvision.datasets.utils import download_and_extract_archive

    data_root = './data/tiny-imagenet-200'
    archive_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    archive_path = './data/tiny-imagenet-200.zip'

    if not os.path.exists(data_root):
        print("📦 Скачиваем Tiny ImageNet-200 (~110MB)...")
        download_and_extract_archive(
            url=archive_url,
            download_root='./data',
            extract_root='./data',
            filename='tiny-imagenet-200.zip'
        )

    # Tiny ImageNet не имеет стандартной структуры, поэтому нужно немного подправить директории
    val_annotations_path = os.path.join(data_root, 'val', 'val_annotations.txt')
    val_images_dir = os.path.join(data_root, 'val', 'images')
    val_target_dir = os.path.join(data_root, 'val_sorted')

    if not os.path.exists(val_target_dir):
        print("🔧 Реформатируем validation-подпапки...")
        os.makedirs(val_target_dir, exist_ok=True)

        with open(val_annotations_path) as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.strip().split('\t')
                img_name = tokens[0]
                class_name = tokens[1]
                class_dir = os.path.join(val_target_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                src = os.path.join(val_images_dir, img_name)
                dst = os.path.join(class_dir, img_name)
                if os.path.exists(src):
                    os.rename(src, dst)

    train_dir = os.path.join(data_root, 'train')
    val_dir = val_target_dir

    transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.480, 0.448, 0.398],
                         [0.277, 0.269, 0.282]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.480, 0.448, 0.398],
                            [0.277, 0.269, 0.282]),
    ])


    train_dataset = ImageFolder(train_dir, transform=transform_train)
    val_dataset = ImageFolder(val_dir, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, train_dataset.classes

# def _load_imagenet(batch_size=128,
#                    data_root='./data/imagenet-1k',
#                    max_samples_per_class=500):
#     import os
#     import random
#     import torch
#     from collections import defaultdict
#     from torchvision import transforms
#     from torchvision.datasets import ImageNet

#     os.makedirs(data_root, exist_ok=True)

#     normalize = transforms.Normalize([0.485, 0.456, 0.406],
#                                      [0.229, 0.224, 0.225])
#     transform_train = transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ])
#     transform_val = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ])

#     # Оригинальные датасеты
#     train_dataset = ImageNet(root=data_root, split='train', transform=transform_train)
#     val_dataset   = ImageNet(root=data_root, split='val',   transform=transform_val)

#     # Если задано ограничение по числу примеров на класс для train
#     if max_samples_per_class is not None:
#         # собираем индексы по классам
#         label_to_indices = defaultdict(list)
#         for idx, (_, label) in enumerate(train_dataset.samples):
#             label_to_indices[label].append(idx)

#         # выбираем случайные подмножества
#         selected_indices = []
#         for label, indices in label_to_indices.items():
#             k = min(max_samples_per_class, len(indices))
#             selected = random.sample(indices, k)
#             selected_indices.extend(selected)

#         # создаём Subset
#         train_dataset = torch.utils.data.Subset(train_dataset, selected_indices)

#     # Даталоадеры
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True,
#         num_workers=4, pin_memory=True
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False,
#         num_workers=4, pin_memory=True
#     )

#     # В случае Subset оригинальные .classes берутся из .dataset
#     classes = (train_dataset.dataset.classes
#                if isinstance(train_dataset, torch.utils.data.Subset)
#                else train_dataset.classes)

#     return train_loader, val_loader, classes


def loader_to_device(loader, device, selected_indices=None):
    if selected_indices is None:
        return list(loader)
    else:
        result = []
        selected_indices = set(selected_indices)
        for idx, (x, y) in enumerate(loader):
            if idx in selected_indices:
                result.append((x, y))
        return result


def get_data_loaders(batch_size, classes=False):
    trainloader, testloader, classes = _load_imagenet(batch_size)
    # trainloader, testloader, classes = _load_cifar(batch_size)

    if classes:
        return trainloader, testloader, classes
    else:    
        return trainloader, testloader

def get_device(pos=None):
    if torch.cuda.is_available():
        if pos is None:
            raise ValueError('Please specify the GPU position')
        else:
            device = torch.device(f'cuda:{pos}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    return device
    

def get_resnet18(device):
    from models import ResNet18
    model = ResNet18()
    model = model.to(device)
    return model


def get_simple_model(device):
    from models import CifarNet
    model = CifarNet()
    model = model.to(device)
    return model

def top_k_accuracy(output, target, ks=(1, 2, 3, 4, 5)):
    with torch.no_grad():
        max_k = max(ks)
        _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = {}
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res[f'top@{k}'] = (correct_k / target.size(0)).item()
        return res

class Logger():
    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.index = []
        self.times = [] 
        self.acc_at_k = []

    def append(self, loss, accuracy, index, elapsed_time=None, acc_at_k=None):
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.index.append(index)
        self.times.append(elapsed_time) 
        self.acc_at_k.append(acc_at_k or {})
    
    def to_dict(self):
        return {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'index': self.index,
            'time': self.times,
            'acc_at_k': self.acc_at_k
        }

import math

def get_warmup_cosine_scheduler(warmup_epochs, total_epochs, base_lr, warmup_lr, min_lr):
    def scheduler(epoch):
        if epoch < warmup_epochs:
            return warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
            return min_lr + (base_lr - min_lr) * cosine_decay
    return scheduler

