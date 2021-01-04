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



		optim_g = optim.RMSprop(generator.parameters(), lr=learning_rate)
		optim_d = optim.RMSprop(discriminator.parameters(), lr=learning_rate)




		losses_g = [] # to store generator loss after each epoch
		losses_d = [] # to store discriminator loss after each epoch
		images = [] # to store images generatd by the generator
		emd_list = []
		epoch_list = []


		# Create batch of latent vectors that we will use to visualize
		#  the progression of the generator
		fixed_noise = torch.randn(64, nz, 1, 1, device=device)



		for epoch in range(epochs):
			loss_g = 0.0
			loss_d = 0.0
			em_distance = 0.0

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
					loss_d1= train_discriminator(discriminator, optim_d, data_real, data_fake)
					loss_d += loss_d1
					em_distance += (-loss_d1)


				data_fake = generator(create_noise(b_size, nz))
				# train the generator
				loss_g1 = train_generator(generator, discriminator, optim_g, data_fake)
				loss_g += loss_g1

			# Check how the generator is doing by saving G's output on fixed_noise
			generated_img = generator(fixed_noise).cpu().detach()
			generated_img = make_grid(generated_img, padding =2, normalize=True)
			save_generator_image(generated_img, f"./6IMAGE_WGAN_CIFAR10/epoch{epoch+1}.png")
			images.append(generated_img)
			epoch_loss_g = loss_g / bi
			epoch_loss_d = loss_d / (k * bi)
			epoch_emd = em_distance / (k * bi)


			losses_g.append(epoch_loss_g)
			losses_d.append(epoch_loss_d)
			epoch_list.append(epoch + 1)
			emd_list.append(epoch_emd)

			print(f"Epoch {epoch + 1} of {epochs}")
			print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}, EMD: {epoch_emd:.8f}")


		fig = plt.figure()
		plt.plot(epoch_list, emd_list, color='black')
		plt.xlabel('Epoch')
		plt.ylabel('EM Distance')
		plt.title('WGAN_CIFAR10_EMD')
		plt.savefig(f"6WGAN_CIFAR10_emd_epoch.png")
		plt.close()


		fig = plt.figure()
		plt.plot(epoch_list, losses_g, color='red', label ='Generator loss' )
		plt.plot(epoch_list, losses_d, color='black', label = 'Critic loss')
		plt.legend(loc='best', prop={'size': 10})
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title('WGAN_CIFAR10_Loss')
		plt.savefig(f"6WGAN_CIFAR10_loss_epoch.png")
		plt.close()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_discriminator(discriminator, optimizer, data_real, data_fake):
	discriminator.train()
	b_size = data_real.size(0) # Get the batch size of data


	optimizer.zero_grad()
	output_real = discriminator(data_real).view(-1) # real data
	output_fake = discriminator(data_fake).view(-1) # fake data
	loss_D = -(torch.mean(output_real) - torch.mean(output_fake))
	loss_D.backward()
	optimizer.step()

	for p in discriminator.parameters():
		p.data.clamp_(-weight_clip, weight_clip)

	return loss_D


def train_generator(generator, discriminator, optimizer, data_fake):
	generator.train()
	b_size = data_fake.size(0)

	optimizer.zero_grad()
	output = discriminator(data_fake).view(-1)
	loss_G = -torch.mean(output)
	loss_G.backward()
	optimizer.step()

	return loss_G



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

		learning_rate=5e-5
		sample_size = 64
		k = 5
		weight_clip = 0.01

		main()