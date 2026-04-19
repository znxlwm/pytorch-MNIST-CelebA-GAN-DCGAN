import os, time, glob
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio.v2 as imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x)) # 修改：F.tanh -> torch.tanh

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x)) # 修改：F.sigmoid -> torch.sigmoid

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1).cuda()    # 修改：直接放GPU，移除 volatile
def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1).cuda() # 修改：直接放GPU，移除 volatile

    G.eval()
    # 修改：使用 no_grad() 替代 volatile=True
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
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
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

def save_checkpoint(epoch, G, D, G_optimizer, D_optimizer, train_hist, save_dir, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'G_optimizer_state_dict': G_optimizer.state_dict(),
        'D_optimizer_state_dict': D_optimizer.state_dict(),
        'train_hist': train_hist
    }
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(G, D, G_optimizer, D_optimizer, load_dir, epoch=None):
    if epoch is not None:
        checkpoint_path = os.path.join(load_dir, f'checkpoint_epoch_{epoch}.pth')
    else:
        checkpoints = glob.glob(os.path.join(load_dir, 'checkpoint_epoch_*.pth'))
        if not checkpoints:
            print("No checkpoints found, starting training from scratch.")
            return 0, None
        checkpoint_path = max(checkpoints, key=os.path.getctime)
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}, starting training from scratch.")
        return 0, None
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
    D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    train_hist = checkpoint.get('train_hist', None)
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming training from epoch {start_epoch + 1}")
    
    return start_epoch, train_hist

# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20
resume_training = True
checkpoint_dir = 'MNIST_DCGAN_results/checkpoints'
tensorboard_log_dir = 'MNIST_DCGAN_results/tensorboard_logs'
checkpoint_save_interval = 1

# data_loader
img_size = 64
transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('MNIST_DCGAN_results'):
    os.mkdir('MNIST_DCGAN_results')
if not os.path.isdir('MNIST_DCGAN_results/Random_results'):
    os.mkdir('MNIST_DCGAN_results/Random_results')
if not os.path.isdir('MNIST_DCGAN_results/Fixed_results'):
    os.mkdir('MNIST_DCGAN_results/Fixed_results')
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
if not os.path.isdir(tensorboard_log_dir):
    os.mkdir(tensorboard_log_dir)

# TensorBoard writer
writer = SummaryWriter(log_dir=tensorboard_log_dir)

# Load checkpoint if resume training
start_epoch = 0
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
num_iter = 0

if resume_training:
    loaded_epoch, loaded_hist = load_checkpoint(G, D, G_optimizer, D_optimizer, checkpoint_dir)
    if loaded_epoch > 0:
        start_epoch = loaded_epoch
        if loaded_hist is not None:
            train_hist = loaded_hist
            num_iter = len(train_hist['D_losses']) * len(train_loader)

print(f"Starting training from epoch {start_epoch + 1}")
start_time = time.time()
for epoch in range(start_epoch, train_epoch):
    current_epoch = epoch + 1
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for x_, _ in train_loader:
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        
        D_result = D(x_).view(-1)
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())
        G_result = G(z_)

        D_result = D(G_result).view(-1)
        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.item())

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())

        G_result = G(z_)
        D_result = D(G_result).view(-1)
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.item())

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    
    avg_D_loss = torch.mean(torch.FloatTensor(D_losses))
    avg_G_loss = torch.mean(torch.FloatTensor(G_losses))

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (current_epoch, train_epoch, per_epoch_ptime, avg_D_loss, avg_G_loss))
    
    # TensorBoard logging
    writer.add_scalar('Loss/Discriminator', avg_D_loss, current_epoch)
    writer.add_scalar('Loss/Generator', avg_G_loss, current_epoch)
    writer.add_scalar('Time/Per_Epoch', per_epoch_ptime, current_epoch)
    writer.add_scalars('Loss/Combined', {
        'Discriminator': avg_D_loss,
        'Generator': avg_G_loss
    }, current_epoch)
    
    # Generate and log images to TensorBoard
    G.eval()
    with torch.no_grad():
        fixed_images = G(fixed_z_)
        fixed_images = (fixed_images + 1) / 2
        writer.add_images('Generated/Fixed_Noise', fixed_images, current_epoch)
        
        random_z = torch.randn((25, 100)).view(-1, 100, 1, 1).cuda()
        random_images = G(random_z)
        random_images = (random_images + 1) / 2
        writer.add_images('Generated/Random_Noise', random_images, current_epoch)
    G.train()
    
    p = 'MNIST_DCGAN_results/Random_results/MNIST_DCGAN_' + str(current_epoch) + '.png'
    fixed_p = 'MNIST_DCGAN_results/Fixed_results/MNIST_DCGAN_' + str(current_epoch) + '.png'
    show_result(current_epoch, save=True, path=p, isFix=False)
    show_result(current_epoch, save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(avg_D_loss)
    train_hist['G_losses'].append(avg_G_loss)
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    
    # Save checkpoint
    if current_epoch % checkpoint_save_interval == 0:
        save_checkpoint(current_epoch, G, D, G_optimizer, D_optimizer, train_hist, checkpoint_dir)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "MNIST_DCGAN_results/generator_param.pkl")
torch.save(D.state_dict(), "MNIST_DCGAN_results/discriminator_param.pkl")
with open('MNIST_DCGAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='MNIST_DCGAN_results/MNIST_DCGAN_train_hist.png')

# Close TensorBoard writer
writer.close()
print("TensorBoard writer closed.")

images = []
for e in range(train_epoch):
    img_name = 'MNIST_DCGAN_results/Fixed_results/MNIST_DCGAN_' + str(e + 1) + '.png'
    if os.path.exists(img_name):
        images.append(imageio.imread(img_name))
if images:
    imageio.mimsave('MNIST_DCGAN_results/generation_animation.gif', images, fps=5)