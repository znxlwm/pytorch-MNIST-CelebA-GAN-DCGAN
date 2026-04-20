import os
from torchvision import datasets, transforms
import torch.utils.data

class DatasetConfig:
    MNIST = {
        'name': 'MNIST',
        'channels': 1,
        'img_size': 64,
        'normalize_mean': (0.5,),
        'normalize_std': (0.5,),
        'download': True,
    }
    
    CELEBA = {
        'name': 'CelebA',
        'channels': 3,
        'img_size': 64,
        'normalize_mean': (0.5, 0.5, 0.5),
        'normalize_std': (0.5, 0.5, 0.5),
        'data_dir': 'data/resized_celebA',
    }
    
    CIFAR10 = {
        'name': 'CIFAR10',
        'channels': 3,
        'img_size': 64,
        'normalize_mean': (0.5, 0.5, 0.5),
        'normalize_std': (0.5, 0.5, 0.5),
        'download': True,
    }

def get_data_loader(dataset_name, batch_size=128, data_root='./data', img_size=64):
    dataset_name = dataset_name.upper()
    
    if dataset_name == 'MNIST':
        return get_mnist_loader(batch_size, data_root, img_size)
    elif dataset_name == 'CELEBA':
        return get_celeba_loader(batch_size, data_root, img_size)
    elif dataset_name == 'CIFAR10':
        return get_cifar10_loader(batch_size, data_root, img_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: MNIST, CELEBA, CIFAR10")

def get_mnist_loader(batch_size, data_root, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    return loader, DatasetConfig.MNIST

def get_celeba_loader(batch_size, data_root, img_size):
    data_dir = os.path.join(data_root, 'resized_celebA')
    
    if not os.path.exists(data_dir):
        raise ValueError(
            f"CelebA dataset not found at {data_dir}. "
            "Please download CelebA and run celebA_data_preprocess.py first."
        )
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    return loader, DatasetConfig.CELEBA

def get_cifar10_loader(batch_size, data_root, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    return loader, DatasetConfig.CIFAR10
