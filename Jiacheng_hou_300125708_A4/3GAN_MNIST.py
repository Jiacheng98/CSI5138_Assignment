import argparse
import os

import torch
from torchtext import data, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math


SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Prevent matplotlib can't run in a non-interactive mode
import matplotlib
matplotlib.use('Agg') 

# Generator

class Generator(nn.Module):
	def __init__(self, nz):
		super(Generator, self).__init__()
		self.nz = nz
		self.main = nn.Sequential(
              nn.Linear(self.nz, 256),
              nn.LeakyReLU(0.2),

              nn.Linear(256, 512),
              nn.LeakyReLU(0.2),

              nn.Linear(512, 1024),
              nn.LeakyReLU(0.2),

              nn.Linear(1024, 784),
              nn.Tanh(),
              )


	def forward(self, x):
			return self.main(x).view(-1, 1, 28, 28)


# Discriminator

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.n_input = 784
		self.main = nn.Sequential(
		        nn.Linear(self.n_input, 1024),
		        nn.LeakyReLU(0.2),
		        nn.Dropout(0.3),

				nn.Linear(1024, 512),
				nn.LeakyReLU(0.2),
				nn.Dropout(0.3),

				nn.Linear(512, 256),
				nn.LeakyReLU(0.2),
				nn.Dropout(0.3),

				nn.Linear(256, 1),
				nn.Sigmoid(),
		)

	def forward(self, x):
				x = x.view(-1, 784)
				return self.main(x)



def main():

		batch_size = 128
		epochs = 200
		sample_size = 64 # sampling a fixed size noise vector
		nz = 128 # latent vector size/noise vector size
		k = 1 # number of steps to apply to the discriminator

		learning_rate=2e-4

		transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
        ])

		train_dataset = torchvision.datasets.MNIST(root='./DATA_MNIST',train=True, transform=transform, download=True)
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



		generator = Generator(nz).to(device)
		discriminator = Discriminator().to(device)


		# print('##### GENERATOR #####')
		# print(generator)
		# print('######################')
		# print('\n##### DISCRIMINATOR #####')
		# print(discriminator)
		# print('######################')



		optim_g = optim.Adam(generator.parameters(), lr=learning_rate,  betas=(0.5, 0.999))
		optim_d = optim.Adam(discriminator.parameters(), lr=learning_rate,  betas=(0.5, 0.999))


		criterion = nn.BCELoss()


		losses_g = [] # to store generator loss after each epoch
		losses_d = [] # to store discriminator loss after each epoch
		images = [] # to store images generatd by the generator

		epoch_list = []

		jsd_losses_g = []
		jsd_losses_d = []



		noise = create_noise(sample_size, nz)


		for epoch in range(epochs):
			loss_g = 0.0
			loss_d = 0.0

			jsd_loss_d = 0.0
			jsd_loss_g = 0.0

			for bi, data in tqdm(enumerate(train_loader), total = int(len(train_dataset) / train_loader.batch_size)):
				image, _ = data
				image = image.to(device)
				b_size = len(image)

				# run the discriminator for k number of steps
				for step in range(k):
					# XXX.backward(): clear from the cache, save memory?
					# Want to reutilize data_fake later, can add '.detach()'
					# When run .backaward(), not clear the intermediate computation
					# Or use .backward(retain_graph = True)
					data_fake = generator(create_noise(b_size, nz)).detach()
					data_real = image
					# train the discriminator
					loss_d1= train_discriminator(discriminator, criterion, optim_d, data_real, data_fake)
					loss_d += loss_d1
					jsd_loss_d += 0.5 * (-loss_d1 + math.log(4))


				data_fake = generator(create_noise(b_size, nz))
				# train the generator
				loss_g1 = train_generator(generator, discriminator, criterion, optim_g, data_fake)
				loss_g += loss_g1
				
			generated_img = generator(noise).cpu().detach()
			generated_img = make_grid(generated_img, padding =2, normalize = True)
			save_generator_image(generated_img, f"./3IMAGE_GAN_MNIST/epoch{epoch+1}.png")
			images.append(generated_img)
			epoch_loss_g = loss_g / bi
			epoch_loss_d = loss_d / bi

			epoch_jsd_loss_d = (jsd_loss_d / bi).item()

			losses_g.append(epoch_loss_g)
			losses_d.append(epoch_loss_d)
			epoch_list.append(epoch + 1)

			jsd_losses_d.append(epoch_jsd_loss_d)
			print(f"Epoch {epoch + 1} of {epochs}")
			print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")
			print(f"Discriminator JSD loss:  {epoch_jsd_loss_d}")


		fig = plt.figure()
		plt.plot(epoch_list, losses_d, color='black', label = "discriminator loss")
		plt.plot(epoch_list, losses_g, color='red', label = "generator loss")
		plt.legend(loc='best', prop={'size': 10})
		plt.xlabel('Epoch')
		plt.ylabel('BCE Loss')
		plt.title('GAN_MNIST_BCE_loss')
		plt.savefig(f"3GAN_MNIST_BCE_loss_epoch.png")
		plt.close()


		fig = plt.figure()
		plt.plot(epoch_list, jsd_losses_d, color='black')
		plt.xlabel('Epoch')
		plt.ylabel('JSD')
		plt.yticks(np.arange(0, 1.05, step=0.1)) 
		plt.title('GAN_MNIST_JSD')
		plt.savefig(f"3GAN_MNIST_JSD_epoch.png")
		plt.close()



def train_discriminator(discriminator, criterion, optimizer, data_real, data_fake):
	discriminator.train()
	b_size = data_real.size(0) # Get the batch size of data
	real_label = label_real(b_size) # Create real labels with batch size
	fake_label = label_fake(b_size)

	optimizer.zero_grad()
	# Train Discriminator: max {log(D(real)) + log(1 - D(G(z)))}
	# Want to minimize loss, so add 'negative sign' to the above function (BCEloss)
	# Maximum the positive value == minimum the negative loss
	output_real = discriminator(data_real) # real data
	loss_real = criterion(output_real, real_label)


	output_fake = discriminator(data_fake) # fake data
	loss_fake = criterion(output_fake, fake_label) 


	loss_real.backward()
	loss_fake.backward()
	# D_loss = -(torch.mean(torch.log(discriminator(data_real))) + torch.mean(torch.log(1- discriminator(data_fake))))
	# D_loss.backward()

	optimizer.step()
	return loss_real + loss_fake


def train_generator(generator, discriminator, criterion, optimizer, data_fake):
	generator.train()
	b_size = data_fake.size(0)
	real_label = label_real(b_size) # Generator wants fake data to have real labels(1)

	optimizer.zero_grad()
	# Train Generator min {log(1-D(G(z)))} (week gradients) == max {log(D(G(z)))} (better)
	output = discriminator(data_fake)
	loss = criterion(output, real_label)
	loss.backward()


	# G_loss = -torch.mean(torch.log(output))
	# G_loss.backward()
	optimizer.step()

	return loss



# to create real labels (1s), size: batch_size
def label_real(size):
	data = torch.ones(size,1)
	return data.to(device)

# to create fake labels (0s)
def label_fake(size):
	data = torch.zeros(size,1)
	return data.to(device)

# function to create the noise vector, size: nz(128)
def create_noise(sample_size, nz):
	return torch.randn(sample_size, nz).to(device)


# to save the images generated by the generator
def save_generator_image(image, path):
    save_image(image, path)


if __name__=="__main__":
      
    
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0", type=str)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    main()