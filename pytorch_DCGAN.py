import os, time, sys
import argparse
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from data_loader import get_data_loader

class Generator(nn.Module):
    def __init__(self, d=128, channels=3):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, channels, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, d=128, channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(channels, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)

def show_result(G, num_epoch, dataset_info, show=False, save=False, path='result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    channels = dataset_info['channels']
    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        if channels == 1:
            ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')
        else:
            ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='DCGAN Training with PyTorch')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'celeba', 'cifar10'],
                        help='Dataset to use: mnist, celeba, or cifar10 (default: mnist)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate (default: 0.0002)')
    parser.add_argument('--train_epoch', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--img_size', type=int, default=64,
                        help='Image size (default: 64)')
    parser.add_argument('--g_dim', type=int, default=128,
                        help='Generator dimension (default: 128)')
    parser.add_argument('--d_dim', type=int, default=128,
                        help='Discriminator dimension (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for storing data (default: ./data)')
    parser.add_argument('--use_lr_decay', action='store_true', default=False,
                        help='Use learning rate decay (default: False)')
    return parser.parse_args()

def get_result_dir(dataset_name):
    """Get the result directory based on dataset name."""
    dir_mapping = {
        'mnist': 'MNIST_DCGAN_results',
        'celeba': 'CelebA_DCGAN_results',
        'cifar10': 'CIFAR10_DCGAN_results'
    }
    return dir_mapping.get(dataset_name.lower(), 'DCGAN_results')

def main():
    args = parse_args()
    
    print(f"Training DCGAN on {args.dataset.upper()} dataset...")
    print(f"Parameters: batch_size={args.batch_size}, lr={args.lr}, epochs={args.train_epoch}")
    print(f"Image size: {args.img_size}, G_dim: {args.g_dim}, D_dim: {args.d_dim}")
    
    train_loader, dataset_info = get_data_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded: {dataset_info['name']}, channels: {dataset_info['channels']}")
    
    G = Generator(d=args.g_dim, channels=dataset_info['channels'])
    D = Discriminator(d=args.d_dim, channels=dataset_info['channels'])
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()
    
    BCE_loss = nn.BCELoss()
    
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    result_dir = get_result_dir(args.dataset)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    if not os.path.isdir(f'{result_dir}/Random_results'):
        os.mkdir(f'{result_dir}/Random_results')
    if not os.path.isdir(f'{result_dir}/Fixed_results'):
        os.mkdir(f'{result_dir}/Fixed_results')
    
    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    num_iter = 0
    
    print('Training start!')
    start_time = time.time()
    for epoch in range(args.train_epoch):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        
        if args.use_lr_decay:
            if (epoch+1) == 11:
                G_optimizer.param_groups[0]['lr'] /= 10
                D_optimizer.param_groups[0]['lr'] /= 10
                print("Learning rate changed!")
            if (epoch+1) == 16:
                G_optimizer.param_groups[0]['lr'] /= 10
                D_optimizer.param_groups[0]['lr'] /= 10
                print("Learning rate changed!")
        
        for x_, _ in train_loader:
            D.zero_grad()
            mini_batch = x_.size()[0]
            
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            
            x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            D_result = D(x_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)
            
            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())
            G_result = G(z_)
            
            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()
            
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()
            
            D_losses.append(D_train_loss.data[0])
            
            G.zero_grad()
            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())
            
            G_result = G(z_)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_)
            G_train_loss.backward()
            G_optimizer.step()
            
            G_losses.append(G_train_loss.data[0])
            
            num_iter += 1
        
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % 
              ((epoch + 1), args.train_epoch, per_epoch_ptime, 
               torch.mean(torch.FloatTensor(D_losses)),
               torch.mean(torch.FloatTensor(G_losses))))
        
        p = f'{result_dir}/Random_results/{args.dataset.upper()}_DCGAN_' + str(epoch + 1) + '.png'
        fixed_p = f'{result_dir}/Fixed_results/{args.dataset.upper()}_DCGAN_' + str(epoch + 1) + '.png'
        show_result(G, (epoch+1), dataset_info, save=True, path=p, isFix=False)
        show_result(G, (epoch+1), dataset_info, save=True, path=fixed_p, isFix=True)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)
    
    print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % 
          (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), args.train_epoch, total_ptime))
    print("Training finish!... save training results")
    torch.save(G.state_dict(), f"{result_dir}/generator_param.pkl")
    torch.save(D.state_dict(), f"{result_dir}/discriminator_param.pkl")
    with open(f'{result_dir}/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)
    
    show_train_hist(train_hist, save=True, path=f'{result_dir}/{args.dataset.upper()}_DCGAN_train_hist.png')
    
    images = []
    for e in range(args.train_epoch):
        img_name = f'{result_dir}/Fixed_results/{args.dataset.upper()}_DCGAN_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(f'{result_dir}/generation_animation.gif', images, fps=5)
    
    print(f"All results saved to {result_dir}/")

if __name__ == '__main__':
    main()
