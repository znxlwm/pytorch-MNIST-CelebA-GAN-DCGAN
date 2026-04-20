import os
import sys
import torch
from torchvision import datasets, transforms

def get_dataset_config(dataset_name):
    configs = {
        'mnist': {
            'channels': 1,
            'img_size': 64,
            'is_gray': True,
            'mean': (0.5,),
            'std': (0.5,)
        },
        'celeba': {
            'channels': 3,
            'img_size': 64,
            'is_gray': False,
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5)
        },
        'cifar10': {
            'channels': 3,
            'img_size': 64,
            'is_gray': False,
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5)
        }
    }
    return configs.get(dataset_name.lower(), configs['cifar10'])

def get_data_loader(dataset_name, batch_size=128, data_dir='./data', img_size=64, num_workers=2):
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
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
        
    elif dataset_name == 'celeba':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        data_dir = os.path.join(data_dir, 'resized_celebA')
        if not os.path.isdir(data_dir):
            sys.stderr.write('Error! CelebA dataset not found at %s\n' % data_dir)
            sys.stderr.write('Please run "celebA_data_preprocess.py" first or use another dataset.\n')
            sys.exit(1)
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
    else:
        raise ValueError('Unknown dataset: %s. Choose from: mnist, celeba, cifar10' % dataset_name)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return data_loader
