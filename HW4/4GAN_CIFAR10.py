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
	def __init__(self):
		super(Generator, self).__init__()
		self.main = nn.Sequential(
		nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
		nn.BatchNorm2d(ngf * 8),
		nn.ReLU(True),
		nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
		nn.BatchNorm2d(ngf * 4),
		nn.ReLU(True),
		nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
		nn.BatchNorm2d(ngf * 2),
		nn.ReLU(True),
		nn.ConvTranspose2d( ngf * 2, nc, 4, 2, 1, bias=False),
		nn.Tanh()
		)

	def forward(self, input):
		return self.main(input)


# Discriminator

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.main = nn.Sequential(
		nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
		nn.BatchNorm2d(ndf * 2),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
		nn.BatchNorm2d(ndf * 4),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
		nn.Sigmoid()
		)

	def forward(self, input):
		return self.main(input)



def main():


		transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5, 0.5),(0.5, 0.5, 0.5)),
        ])

		train_dataset = torchvision.datasets.CIFAR10(root='./DATA_CIFAR10',train=True, transform=transform, download=True)
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


		generator = Generator().to(device)
		generator.apply(weights_init)
		discriminator = Discriminator().to(device)
		discriminator.apply(weights_init)


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


		# Create batch of latent vectors that we will use to visualize
		#  the progression of the generator
		fixed_noise = torch.randn(64, nz, 1, 1, device=device)



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
					# detach: not be involved in back-propogation???
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

			# Check how the generator is doing by saving G's output on fixed_noise
			generated_img = generator(fixed_noise).cpu().detach()
			generated_img = make_grid(generated_img, padding =2, normalize=True)
			save_generator_image(generated_img, f"./4IMAGE_GAN_CIFAR10/epoch{epoch+1}.png")
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
		plt.title('GAN_CIFAR10_BCE_loss')
		plt.savefig(f"4GAN_CIFAR10_BCE_loss_epoch.png")
		plt.close()


		fig = plt.figure()
		plt.plot(epoch_list, jsd_losses_d, color='black')
		plt.xlabel('Epoch')
		plt.yticks(np.arange(0, 1.05, step=0.1)) 
		plt.ylabel('JS Divergence')
		plt.title('GAN_CIFAR10_JSD')
		plt.savefig(f"4GAN_CIFAR10_JSD_epoch.png")
		plt.close()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_discriminator(discriminator, criterion, optimizer, data_real, data_fake):
	discriminator.train()
	b_size = data_real.size(0) # Get the batch size of data
	real_label = label_real(b_size) # Create real labels with batch size
	fake_label = label_fake(b_size)


	optimizer.zero_grad()

	output_real = discriminator(data_real).view(-1) # real data
	loss_real = criterion(output_real, real_label)
	# jsd_loss_real = js_div(output_real, real_label)

	output_fake = discriminator(data_fake).view(-1) # fake data
	loss_fake = criterion(output_fake, fake_label) 
	# jsd_loss_fake = js_div(output_fake, fake_label)

	loss_real.backward()
	loss_fake.backward()
	optimizer.step()

	return loss_real+loss_fake


def train_generator(generator, discriminator, criterion, optimizer, data_fake):
	generator.train()
	b_size = data_fake.size(0)
	real_label = label_real(b_size) # Generator wants fake data to have real labels(1)

	optimizer.zero_grad()

	output = discriminator(data_fake).view(-1)
	loss = criterion(output, real_label)
	# jsd_loss = js_div(output, real_label)

	loss.backward()
	optimizer.step()

	return loss



# to create real labels (1s), size: batch_size
def label_real(size):
	# Establish convention for real and fake labels during training
	real_label = 1.
	data = torch.full((size,), real_label, dtype=torch.float, device=device)
	return data

# to create fake labels (0s)
def label_fake(size):
	fake_label = 0.
	data = torch.full((size,), fake_label, dtype=torch.float, device=device)
	return data

# Generate batch of latent vectors
def create_noise(sample_size, nz):
	return torch.randn(sample_size, nz, 1, 1, device=device)


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


		batch_size = 128
		image_size = 32
		nc = 3 # no. of corlor channels
		nz = 128 # latent vector length
		ngf = 32 # depth of feature maps carried through the generator
		ndf = 32 # depth of feature maps propagated through the discriminator
		epochs = 200

		learning_rate=2e-4
		sample_size = 64
		k = 1

		main()