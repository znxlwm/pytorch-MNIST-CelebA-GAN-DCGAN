import os, time, sys, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_data_loader, get_dataset_config
from models import Generator, Discriminator

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1).cuda()

def show_result(G, num_epoch, dataset_config, show=False, save=False, path='result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1).cuda()

    G.eval()
    with torch.no_grad():
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

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        if dataset_config['is_gray']:
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
    parser = argparse.ArgumentParser(description='DCGAN Training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'celeba', 'cifar10'],
                        help='Dataset to use (mnist, celeba, cifar10)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for dataset storage')
    parser.add_argument('--img_size', type=int, default=64,
                        help='Image size for training')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_name = args.dataset
    batch_size = args.batch_size
    lr = args.lr
    train_epoch = args.epochs
    data_dir = args.data_dir
    img_size = args.img_size
    num_workers = args.num_workers

    dataset_config = get_dataset_config(dataset_name)
    print('Dataset: %s' % dataset_name)
    print('Image size: %d x %d' % (img_size, img_size))
    print('Channels: %d' % dataset_config['channels'])

    print('Loading data...')
    train_loader = get_data_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        data_dir=data_dir,
        img_size=img_size,
        num_workers=num_workers
    )
    print('Data loaded successfully! Number of batches: %d' % len(train_loader))

    G = Generator(128, output_channels=dataset_config['channels'])
    D = Discriminator(128, input_channels=dataset_config['channels'])
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()

    BCE_loss = nn.BCELoss()

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    results_dir = '%s_DCGAN_results' % dataset_name.upper()
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    if not os.path.isdir('%s/Random_results' % results_dir):
        os.mkdir('%s/Random_results' % results_dir)
    if not os.path.isdir('%s/Fixed_results' % results_dir):
        os.mkdir('%s/Fixed_results' % results_dir)

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    num_iter = 0

    print('Training start!')
    start_time = time.time()
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        
        if dataset_name == 'celeba':
            if (epoch + 1) == 11:
                G_optimizer.param_groups[0]['lr'] /= 10
                D_optimizer.param_groups[0]['lr'] /= 10
                print("Learning rate changed!")
            if (epoch + 1) == 16:
                G_optimizer.param_groups[0]['lr'] /= 10
                D_optimizer.param_groups[0]['lr'] /= 10
                print("Learning rate changed!")

        for x_, _ in train_loader:
            D.zero_grad()

            mini_batch = x_.size()[0]

            y_real_ = torch.ones(mini_batch).cuda()
            y_fake_ = torch.zeros(mini_batch).cuda()
            x_ = x_.cuda()
            
            D_result = D(x_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).cuda()
            G_result = G(z_)

            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.item())

            G.zero_grad()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).cuda()

            G_result = G(z_)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_)
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.item())

            num_iter += 1

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), train_epoch, per_epoch_ptime,
            torch.mean(torch.FloatTensor(D_losses)),
            torch.mean(torch.FloatTensor(G_losses))))
        
        p = '%s/Random_results/%s_DCGAN_%d.png' % (results_dir, dataset_name.upper(), epoch + 1)
        fixed_p = '%s/Fixed_results/%s_DCGAN_%d.png' % (results_dir, dataset_name.upper(), epoch + 1)
        show_result(G, epoch + 1, dataset_config, save=True, path=p, isFix=False)
        show_result(G, epoch + 1, dataset_config, save=True, path=fixed_p, isFix=True)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
        torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])),
        train_epoch, total_ptime))
    print("Training finish!... save training results")
    
    torch.save(G.state_dict(), "%s/generator_param.pkl" % results_dir)
    torch.save(D.state_dict(), "%s/discriminator_param.pkl" % results_dir)
    with open('%s/train_hist.pkl' % results_dir, 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path='%s/%s_DCGAN_train_hist.png' % (results_dir, dataset_name.upper()))

    images = []
    for e in range(train_epoch):
        img_name = '%s/Fixed_results/%s_DCGAN_%d.png' % (results_dir, dataset_name.upper(), e + 1)
        images.append(imageio.imread(img_name))
    imageio.mimsave('%s/generation_animation.gif' % results_dir, images, fps=5)
    print("Results saved to: %s" % results_dir)

if __name__ == '__main__':
    main()
