import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image


mnist_path = './mnist_data'
fashionmnist_path = './fashionmnist_data'
# TODO: need a path to result
if not os.path.exists(mnist_path):
    os.mkdir(mnist_path)

if not os.path.exists(fashionmnist_path):
    os.mkdir(fashionmnist_path)


def load_data(batch_size=100, use_mnist=True):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if use_mnist:
        train_set = datasets.MNIST(root=mnist_path, train=True, transform=trans, download=True)
    else:
        train_set = datasets.FashionMNIST(root=fashionmnist_path, train=True, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)
    return train_loader


def create_and_save_image(generator, test_noise, num_fig):
    test_images = generator(test_noise)
    test_images = test_images.view(test_images.shape[0], *(1, 28, 28))
    save_image(test_images, 'images/%d.png' % num_fig, nrow=6, normalize=True)


def train_discriminator(discriminator, criterion, d_optimizer, real_images, real_labels, fake_images, fake_labels):
    discriminator.zero_grad()
    real_images = real_images.view(-1, 28*28)
    outputs = discriminator(real_images)
    real_loss = criterion(outputs, real_labels)
    real_score = outputs

    outputs = discriminator(fake_images)
    fake_loss = criterion(outputs, fake_labels)
    fake_score = outputs

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()

    # Clip weights of discriminator
    for p in discriminator.parameters():
        p.data.clamp_(-0.01, 0.01)

    return d_loss, real_score, fake_score


def train_generator(generator, criterion, g_optimizer, discriminator_outputs, real_labels):
    generator.zero_grad()
    g_loss = criterion(discriminator_outputs, real_labels)
    g_loss.backward()
    g_optimizer.step()
    return g_loss
