import os
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio.v2 as imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# G(z)
class generator(nn.Module):
    def __init__(self, input_size=32, n_class=10):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x)) # 这里的 F.tanh 在旧版，推荐用 torch.tanh
        return x

class discriminator(nn.Module):
    def __init__(self, input_size=32, n_class=10):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = torch.sigmoid(self.fc4(x))
        return x

# 固定噪声用于观察生成效果
fixed_z_ = torch.randn((5 * 5, 100)).cuda()

def show_result(num_epoch, show=False, save=False, path='result.png', isFix=False):
    G.eval()
    with torch.no_grad():
        if isFix:
            test_images = G(fixed_z_)
        else:
            z_ = torch.randn((5 * 5, 100)).cuda()
            test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5 * 5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    
    if save:
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
    plt.xlabel('Epoch')
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

# 训练参数
batch_size = 128
lr = 0.0002
train_epoch = 20

# 数据加载 - 关键点：MNIST是灰度图，Normalize 只能有一个维度
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# 网络初始化
G = generator(input_size=100, n_class=28*28)
D = discriminator(input_size=28*28, n_class=1)
G.cuda()
D.cuda()

# 损失函数与优化器
BCE_loss = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

# 结果保存路径
folders = ['MNIST_GAN_results/Random_results', 'MNIST_GAN_results/Fixed_results']
for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)

train_hist = {'D_losses': [], 'G_losses': []}

for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    for x_, _ in train_loader:
        # 训练判别器 D
        D.zero_grad()
        mini_batch = x_.size()[0]
        
        x_ = x_.view(-1, 28 * 28).cuda()
        y_real_ = torch.ones(mini_batch, 1).cuda()
        y_fake_ = torch.zeros(mini_batch, 1).cuda()

        # 真图损失
        D_result = D(x_)
        D_real_loss = BCE_loss(D_result, y_real_)

        # 假图损失
        z_ = torch.randn((mini_batch, 100)).cuda()
        G_result = G(z_)
        D_result = D(G_result.detach()) # 训练D时，切断G的梯度
        D_fake_loss = BCE_loss(D_result, y_fake_)

        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        D_optimizer.step()
        D_losses.append(D_train_loss.item())

        # 训练生成器 G
        G.zero_grad()
        z_ = torch.randn((mini_batch, 100)).cuda()
        G_result = G(z_)
        D_result = D(G_result)
        G_train_loss = BCE_loss(D_result, y_real_) # 希望判别器把假图看作真图

        G_train_loss.backward()
        G_optimizer.step()
        G_losses.append(G_train_loss.item())

    # 打印进度
    mean_d_loss = torch.mean(torch.FloatTensor(D_losses))
    mean_g_loss = torch.mean(torch.FloatTensor(G_losses))
    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, mean_d_loss, mean_g_loss))

    # 保存图片
    p = 'MNIST_GAN_results/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    fixed_p = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=p, isFix=False)
    show_result((epoch+1), save=True, path=fixed_p, isFix=True)

    train_hist['D_losses'].append(mean_d_loss)
    train_hist['G_losses'].append(mean_g_loss)

print("Training finish! Saving results...")
torch.save(G.state_dict(), "MNIST_GAN_results/generator_param.pkl")
torch.save(D.state_dict(), "MNIST_GAN_results/discriminator_param.pkl")
with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')

# 生成动画
images = []
for e in range(train_epoch):
    img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)