import os
import time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.optim as optim

class DCGANTrainer:
    def __init__(self, generator, discriminator, train_loader, dataset_config,
                 batch_size=128, lr=0.0002, train_epoch=20,
                 results_dir=None, use_cuda=True):
        self.G = generator
        self.D = discriminator
        self.train_loader = train_loader
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.lr = lr
        self.train_epoch = train_epoch
        self.use_cuda = use_cuda
        
        dataset_name = dataset_config['name']
        if results_dir is None:
            self.results_dir = f'{dataset_name}_DCGAN_results'
        else:
            self.results_dir = results_dir
        
        self.channels = dataset_config['channels']
        self._setup_results_dir()
        self._setup_optimizers()
        self._setup_fixed_noise()
        
    def _setup_results_dir(self):
        dirs = [
            self.results_dir,
            os.path.join(self.results_dir, 'Random_results'),
            os.path.join(self.results_dir, 'Fixed_results')
        ]
        for d in dirs:
            if not os.path.isdir(d):
                os.makedirs(d)
    
    def _setup_optimizers(self):
        self.BCE_loss = nn.BCELoss()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss.cuda()
    
    def _setup_fixed_noise(self):
        self.fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
        if self.use_cuda:
            self.fixed_z_ = self.fixed_z_.cuda()
    
    def show_result(self, num_epoch, show=False, save=False, path='result.png', isFix=False):
        z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
        if self.use_cuda:
            z_ = z_.cuda()

        self.G.eval()
        with torch.no_grad():
            if isFix:
                test_images = self.G(self.fixed_z_)
            else:
                test_images = self.G(z_)
        self.G.train()

        size_figure_grid = 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(5*5):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            if self.channels == 1:
                ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')
            else:
                ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

        label = 'Epoch {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')
        
        if save:
            plt.savefig(path)
        if show:
            plt.show()
        else:
            plt.close()
    
    def show_train_hist(self, hist, show=False, save=False, path='Train_hist.png'):
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
    
    def train(self, lr_decay_epochs=[11, 16]):
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []
        
        print('Training start!')
        start_time = time.time()
        
        for epoch in range(self.train_epoch):
            D_losses = []
            G_losses = []
            
            if (epoch+1) in lr_decay_epochs:
                self.G_optimizer.param_groups[0]['lr'] /= 10
                self.D_optimizer.param_groups[0]['lr'] /= 10
                print("learning rate change!")
            
            epoch_start_time = time.time()
            
            for x_, _ in self.train_loader:
                mini_batch = x_.size()[0]
                
                y_real_ = torch.ones(mini_batch)
                y_fake_ = torch.zeros(mini_batch)
                
                if self.use_cuda:
                    x_ = x_.cuda()
                    y_real_ = y_real_.cuda()
                    y_fake_ = y_fake_.cuda()
                
                self.D.zero_grad()
                D_result = self.D(x_).squeeze()
                D_real_loss = self.BCE_loss(D_result, y_real_)
                
                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                if self.use_cuda:
                    z_ = z_.cuda()
                G_result = self.G(z_)
                
                D_result = self.D(G_result).squeeze()
                D_fake_loss = self.BCE_loss(D_result, y_fake_)
                
                D_train_loss = D_real_loss + D_fake_loss
                D_train_loss.backward()
                self.D_optimizer.step()
                
                D_losses.append(D_train_loss.item())
                
                self.G.zero_grad()
                z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
                if self.use_cuda:
                    z_ = z_.cuda()
                G_result = self.G(z_)
                D_result = self.D(G_result).squeeze()
                G_train_loss = self.BCE_loss(D_result, y_real_)
                
                G_train_loss.backward()
                self.G_optimizer.step()
                
                G_losses.append(G_train_loss.item())
            
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            
            mean_d_loss = torch.mean(torch.FloatTensor(D_losses))
            mean_g_loss = torch.mean(torch.FloatTensor(G_losses))
            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % 
                  ((epoch + 1), self.train_epoch, per_epoch_ptime, mean_d_loss, mean_g_loss))
            
            p = os.path.join(self.results_dir, 'Random_results', f"{self.dataset_config['name']}_DCGAN_{epoch + 1}.png")
            fixed_p = os.path.join(self.results_dir, 'Fixed_results', f"{self.dataset_config['name']}_DCGAN_{epoch + 1}.png")
            
            self.show_result(epoch+1, save=True, path=p, isFix=False)
            self.show_result(epoch+1, save=True, path=fixed_p, isFix=True)
            
            train_hist['D_losses'].append(mean_d_loss)
            train_hist['G_losses'].append(mean_g_loss)
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        
        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)
        
        print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % 
              (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), 
               self.train_epoch, total_ptime))
        print("Training finish!... save training results")
        
        torch.save(self.G.state_dict(), os.path.join(self.results_dir, 'generator_param.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.results_dir, 'discriminator_param.pkl'))
        with open(os.path.join(self.results_dir, 'train_hist.pkl'), 'wb') as f:
            pickle.dump(train_hist, f)
        
        self.show_train_hist(train_hist, save=True, 
                             path=os.path.join(self.results_dir, f"{self.dataset_config['name']}_DCGAN_train_hist.png"))
        
        self._create_animation()
        
        return train_hist
    
    def _create_animation(self):
        images = []
        for e in range(self.train_epoch):
            img_name = os.path.join(self.results_dir, 'Fixed_results', 
                                    f"{self.dataset_config['name']}_DCGAN_{e + 1}.png")
            images.append(imageio.imread(img_name))
        imageio.mimsave(os.path.join(self.results_dir, 'generation_animation.gif'), images, fps=5)
