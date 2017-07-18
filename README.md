# pytorch-MNIST-GAN
Pytorch implementation of Generative Adversarial Networks (GAN) for MNIST dataset.

## Resutls
* Generate using fixed noise (fixed_z_)
![Generation](MNIST_GAN_results/generation_animation.gif?raw=true)

* MNIST vs Generated images
<table align='center'>
<tr align='center'>
<td> MNIST </td>
<td> Generated images </td>
</tr>
<tr>
<td><img src = ''>
<td><img src = 'MNIST_GAN_resutls/MNIST_GAN_100.png'>
</tr>
<tr>
<td> stdev = 1.0 </td>
<td><img src = 'results/GAN_1d_gaussian_mu_-1_sigma_1.png' height = '300px'>
<td><img src = 'results/GAN_1d_gaussian_mu_1_sigma_1.png' height = '300px'>
</tr>
<tr>
<td> stdev = 2.0 </td>
<td><img src = 'results/GAN_1d_gaussian_mu_-1_sigma_2.png' height = '300px'>
<td><img src = 'results/GAN_1d_gaussian_mu_1_sigma_2.png' height = '300px'>
</tr>
</table>

## Development Environment
* Ubuntu 14.04 LTS
* Python 2.7.6
* cuda 8.0
* pytorch 0.1.12
* torchvision 0.1.8
* matplotlib 1.3.1
* imageio 2.2.0


