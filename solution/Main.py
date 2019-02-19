import sys
sys.path.append('../')  # ugly dirtyfix for imports to work
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
from Introduction_to_pytorch_and_GAN.solution.Utils import *
from Introduction_to_pytorch_and_GAN.solution.Generator import *
from Introduction_to_pytorch_and_GAN.solution.Discriminator import *


if __name__ == '__main__':
    # Initialize some variables regarding the models

    # These variables have great impact on the models performance. Play around and see if you can some interesting
    # combination that gives you cool results. For some hints look at the last slide in the PowerPoint presentation.
    # TODO: Initialize the models variables
    batch_size = 100
    output_dim = 28*28
    z_dim = 100
    lr = 0.0005

    # Initialize the model
    # TODO: Initialize the models
    generator = Generator(z_dim=z_dim, output_dim=output_dim)
    discriminator = Discriminator(input_dim=output_dim)

    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)

    # Printing a summary of the models
    summary(discriminator, input_size=(batch_size, output_dim))
    summary(generator, input_size=(batch_size, z_dim))

    # Initializing the loss function and optimizer
    criterion = nn.BCELoss()
    # Here you should read the pytorch documentation. There is many different optimizers you can choose from, like
    # SGD, RMSprop and Adam. I recommend RMSprop for this task, but you can play around here and see the difference in
    # the generated images.
    # TODO: Initialize the optimizer for the models
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)
    g_optimizer = optim.RMSprop(generator.parameters(), lr=lr)

    # Generating a fixed vector with noise to be used for test images
    num_test_samples = 36
    test_noise = Variable(torch.randn(num_test_samples, z_dim))

    # Loading and preparing training data
    # TODO: NB! When you have tested MNIST for a while you can try Fashion MNIST by setting use_mnist to false
    train_loader = load_data(batch_size=batch_size, use_mnist=True)
    real_labels = Variable(torch.ones(batch_size, 1))
    fake_labels = Variable(torch.zeros(batch_size, 1))

    # Sets some training parameters
    num_epochs = 5
    num_batches = len(train_loader)
    # These variables can be played with to adjust the ratio of generator training and discriminator training
    n_g = 1
    n_d = 1

    # Initialize some printing variables
    print_every = 100
    num_fig = 0
    g_loss = 0
    d_loss = 0
    real_score = 0
    fake_score = 0

    # Creating an image before doing any training
    create_and_save_image(generator, test_noise, num_fig)

    print("################################################################")
    print("Start training for %d epochs" % num_epochs)
    print('Total training batches: {}'.format(len(train_loader)))
    print("################################################################")
    for epoch in range(num_epochs):
        for n, (images, _) in enumerate(train_loader):
            images = Variable(images)

            # Train the discriminator for n_d iterations
            for i in range(n_d):
                noise = Variable(torch.randn(batch_size, z_dim))
                fake_images = generator(noise).detach()
                temp_loss, temp_real_score, temp_fake_score = train_discriminator(discriminator, criterion, d_optimizer,
                                                                          images, real_labels, fake_images, fake_labels)
                d_loss += temp_loss
                real_score += temp_real_score
                fake_score += temp_fake_score

            # Training the generator for n_g iterations
            for i in range(n_g):
                noise = Variable(torch.randn(batch_size, z_dim))
                fake_images = generator(noise)
                outputs = discriminator(fake_images)
                g_loss += train_generator(generator, criterion, g_optimizer, outputs, real_labels)

            if n % print_every == 0 and (n != 0 or epoch != 0):
                num_fig += print_every
                create_and_save_image(generator, test_noise, num_fig)

                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                      % (epoch+1, num_epochs, n, num_batches, d_loss.data/(print_every*n_d),
                         g_loss.data/(print_every*n_g), real_score.data.mean()/(print_every*n_d),
                         fake_score.data.mean()/(print_every*n_d)))
                # Resets the printing variables
                g_loss = 0
                d_loss = 0
                real_score = 0
                fake_score = 0
    print("Done training!")
