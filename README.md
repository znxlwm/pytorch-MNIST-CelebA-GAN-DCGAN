# pytorch-MNIST-GAN
Pytorch implementation of Generative Adversarial Networks (GAN) for MNIST dataset.
Full paper: 

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
<td><img src = 'MNIST_GAN_results/raw_MNIST.png'>
<td><img src = 'MNIST_GAN_results/MNIST_GAN_100.png'>
</tr>
</table>

* Training loss

![Loss](MNIST_GAN_results/MNIST_GAN_train_hist.png)

## Development Environment
* Ubuntu 14.04 LTS
* Python 2.7.6
* cuda 8.0
* pytorch 0.1.12
* torchvision 0.1.8
* matplotlib 1.3.1
* imageio 2.2.0

## Reference
Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
(Full paper: http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
