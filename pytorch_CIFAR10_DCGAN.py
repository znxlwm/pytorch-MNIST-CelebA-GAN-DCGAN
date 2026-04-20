import torch
from data_loader import get_data_loader
from models import Generator, Discriminator
from trainer import DCGANTrainer

def main():
    batch_size = 128
    lr = 0.0002
    train_epoch = 20
    img_size = 64
    use_cuda = torch.cuda.is_available()
    
    print("=" * 50)
    print("CIFAR-10 DCGAN Training")
    print("=" * 50)
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {train_epoch}")
    print(f"Image size: {img_size}")
    print(f"CUDA available: {use_cuda}")
    print("=" * 50)
    
    train_loader, dataset_config = get_data_loader(
        dataset_name='CIFAR10',
        batch_size=batch_size,
        data_root='./data',
        img_size=img_size
    )
    
    print(f"Dataset: {dataset_config['name']}")
    print(f"Channels: {dataset_config['channels']}")
    print(f"Number of batches: {len(train_loader)}")
    print("=" * 50)
    
    G = Generator(d=128, channels=dataset_config['channels'])
    D = Discriminator(d=128, channels=dataset_config['channels'])
    
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    
    trainer = DCGANTrainer(
        generator=G,
        discriminator=D,
        train_loader=train_loader,
        dataset_config=dataset_config,
        batch_size=batch_size,
        lr=lr,
        train_epoch=train_epoch,
        use_cuda=use_cuda
    )
    
    train_hist = trainer.train(lr_decay_epochs=[11, 16])
    
    print("=" * 50)
    print("Training completed successfully!")
    print("Results saved in: CIFAR10_DCGAN_results/")
    print("=" * 50)
    
    return train_hist

if __name__ == '__main__':
    main()
