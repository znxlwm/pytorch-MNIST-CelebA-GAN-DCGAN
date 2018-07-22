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

# G(z)
class generator(nn.Module):
    def __init__(self, input_size=32, output_size = 10):
        # Initialisation
        super(generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, output_size)

    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.tanh(self.fc4(x))
        return x

class discriminator(nn.Module):
    def __init__(self, input_size=32, n_class=10):
        # Initialisation
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
        x = F.sigmoid(self.fc4(x))
        return x

fixed_z_ = torch.randn((5 * 5, 100))    # fixed noise
fixed_z_ = Variable(fixed_z_.cuda())

def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100))
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
        ax[i, j].imshow(test_images[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')

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

# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 100
PATH = './MNIST_GAN_results'
# data_loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G = generator(input_size=100, output_size=28*28).cuda()
D = discriminator(input_size=28*28, n_class=1).cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

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
    x = input.view(-1, 28 * 28)
    x = Variable(x.cuda())

    D_real_result = D(x)
    D_real_loss = BCE_loss(D_real_result, y_real)

    # Calculate loss for generated sample
    z = torch.randn((mini_batch_size, 100))
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
    z = torch.randn((mini_batch_size, 100))
    y = torch.ones(mini_batch_size)     # Attempting to be real
    z, y = Variable(z.cuda()), Variable(y.cuda())

    # Calculate loss for generator
    # Comparing discriminator's prediction with ones (ie, real)
    G_result = G(z)
    D_result = D(G_result)
    G_train_loss = BCE_loss(D_result, y)

    # Propogate loss backwards and return loss
    G_train_loss.backward()
    G_optimizer.step()
    return G_train_loss.item()

def train():
    train_hist = {'D_losses':[],'G_losses':[]}

    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        for x_, _ in train_loader:
            mini_batch_size = x_.size()[0]
            D_loss = train_discriminator(x_,mini_batch_size)
            D_losses.append(D_loss)

            G_loss = train_generator(mini_batch_size)
            G_losses.append(G_loss)

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch + 1), train_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
        p = PATH+'/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        fixed_p = PATH+'/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        show_result((epoch+1), save=True, path=p, isFix=False)
        show_result((epoch+1), save=True, path=fixed_p, isFix=True)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))


    print("Training complete. Saving models")
    torch.save(G.state_dict(), PATH+"/generator_param_"+str(train_epoch)+".pkl")
    torch.save(D.state_dict(), PATH +"/discriminator_param_"+str(train_epoch)+".pkl")

    with open(PATH+'/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path=PATH+'/MNIST_GAN_train_hist.png')

    images = []
    for e in range(train_epoch):
        img_name = PATH+'/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(PATH+'/generation_animation.gif', images, fps=5)

if __name__ == '__main__':
    train()
