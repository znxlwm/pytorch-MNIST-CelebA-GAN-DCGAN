import os
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

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
        # x = F.relu(self.deconv1(input))
        input = input.view(-1,100,1,1)
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

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
        # input = input.view(-1,28,28)
        # print (input.shape)
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        temp = self.conv5(x)
        x = F.sigmoid(temp)

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 100
PATH = './MNIST_DCGAN_results'
NOISE_SIZE = 100

fixed_z_ = torch.randn((5 * 5, NOISE_SIZE))    # fixed noise
fixed_z_ = Variable(fixed_z_.cuda())

def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, NOISE_SIZE))
    z_ = Variable(z_.cuda())

    G.eval()
    test_images = G(fixed_z_) if isFix else G(z_)

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
        ax[i, j].imshow(test_images[k, :].cpu().data.view(64, 64).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    print ("Saved to :",path)
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

# data_loader
img_size = 64
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

print ("Loading data")
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)
print ("Loaded data")

# network
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()


print ("Created models")
# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir(PATH):
    os.mkdir(PATH)
if not os.path.isdir(PATH+'/Random_results'):
    os.mkdir(PATH+'/Random_results')
if not os.path.isdir(PATH+'/Fixed_results'):
    os.mkdir(PATH+'/Fixed_results')

def train_discriminator(input,mini_batch_size):
    D.zero_grad()

    y_real = torch.ones(mini_batch_size)  # D(real) = 1
    y_fake = torch.zeros(mini_batch_size) # D(fake) = 0
    y_real, y_fake =  Variable(y_real.cuda()), Variable(y_fake.cuda())

    # Calculate loss for real sample
    # x = input.view(-1, 28, 28)
    x = Variable(input.cuda())

    D_real_result = D(x).view((-1))
    D_real_loss = BCE_loss(D_real_result, y_real)

    # Calculate loss for generated sample
    z = torch.randn((mini_batch_size, NOISE_SIZE))
    z = Variable(z.cuda())
    G_result = G(z) # Generator's result

    D_fake_result = D(G_result)
    D_fake_loss = BCE_loss(D_fake_result, y_fake)

    # Calculating total loss
    D_train_loss = D_real_loss + D_fake_loss

    # Propogate loss backwards and return loss
    D_train_loss.backward()
    D_optimizer.step()
    return D_train_loss.item()

def train_generator(mini_batch_size):
    G.zero_grad()

    # Generate z with random values
    z = torch.randn((mini_batch_size, NOISE_SIZE))
    y = torch.ones(mini_batch_size)     # Attempting to be real
    z, y = Variable(z.cuda()), Variable(y.cuda())

    # Calculate loss for generator
    # Comparing discriminator's prediction with ones (ie, real)
    G_result = G(z)
    D_result = D(G_result).view((-1))
    G_train_loss = BCE_loss(D_result, y)

    # Propogate loss backwards and return loss
    G_train_loss.backward()
    G_optimizer.step()
    return G_train_loss.item()

def save_models(train_hist):
    torch.save(G.state_dict(), PATH+"/generator_param_"+str(train_epoch)+".pkl")
    torch.save(D.state_dict(), PATH +"/discriminator_param_"+str(train_epoch)+".pkl")

    with open(PATH+'/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

def save_gif():
    images = []
    for e in range(train_epoch):
        img_name = PATH+'/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(PATH+'/generation_animation.gif', images, fps=5)

def train():
    train_hist = {'D_losses':[],'G_losses':[]}
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        iter_lim = 0
        for x_, _ in train_loader:
            iter_lim+=1
            if iter_lim == 20:
                break
            mini_batch_size = x_.size()[0]
            D_loss = train_discriminator(x_,mini_batch_size)
            D_losses.append(D_loss)

            G_loss = train_generator(mini_batch_size)
            G_losses.append(G_loss)
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), train_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
        p = PATH+'/Random_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
        fixed_p = PATH+'/Fixed_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
        show_result((epoch+1), save=True, path=p, isFix=False)
        show_result((epoch+1), save=True, path=fixed_p, isFix=True)

        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))


    print("Training complete. Saving.")
    save_models(train_hist)
    show_train_hist(train_hist, save=True, path=PATH+'/MNIST_GAN_train_hist.png')
    save_gif()

if __name__ == '__main__':
    train()
