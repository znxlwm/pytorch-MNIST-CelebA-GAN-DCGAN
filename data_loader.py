import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(dataset_name, batch_size=128, img_size=64, data_dir='./data', num_workers=4, pin_memory=True):
    """
    Unified data loader for MNIST, CelebA, and CIFAR-10 datasets.
    
    Args:
        dataset_name (str): Name of the dataset ('mnist', 'celeba', 'cifar10')
        batch_size (int): Batch size for training
        img_size (int): Target image size after resizing
        data_dir (str): Directory to store/load data
        num_workers (int): Number of workers for DataLoader
        pin_memory (bool): Whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader: PyTorch DataLoader for the specified dataset
        dict: Dataset information (channels, img_size, etc.)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        return _get_mnist_loader(batch_size, img_size, data_dir, num_workers, pin_memory)
    elif dataset_name == 'celeba':
        return _get_celeba_loader(batch_size, img_size, data_dir, num_workers, pin_memory)
    elif dataset_name == 'cifar10':
        return _get_cifar10_loader(batch_size, img_size, data_dir, num_workers, pin_memory)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from 'mnist', 'celeba', 'cifar10'")

def _get_mnist_loader(batch_size, img_size, data_dir, num_workers, pin_memory):
    """Data loader for MNIST dataset."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    dataset_info = {
        'name': 'mnist',
        'channels': 1,
        'img_size': img_size,
        'num_classes': 10
    }
    
    return data_loader, dataset_info

def _get_celeba_loader(batch_size, img_size, data_dir, num_workers, pin_memory):
    """Data loader for CelebA dataset."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    celeba_dir = os.path.join(data_dir, 'resized_celebA')
    
    if not os.path.exists(celeba_dir):
        raise FileNotFoundError(
            f"CelebA dataset not found at {celeba_dir}. "
            f"Please download CelebA dataset and preprocess it using celebA_data_preprocess.py"
        )
    
    dataset = datasets.ImageFolder(
        root=celeba_dir,
        transform=transform
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    dataset_info = {
        'name': 'celeba',
        'channels': 3,
        'img_size': img_size,
        'num_classes': 1
    }
    
    return data_loader, dataset_info

def _get_cifar10_loader(batch_size, img_size, data_dir, num_workers, pin_memory):
    """Data loader for CIFAR-10 dataset with automatic download and preprocessing."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    dataset_info = {
        'name': 'cifar10',
        'channels': 3,
        'img_size': img_size,
        'num_classes': 10
    }
    
    return data_loader, dataset_info
