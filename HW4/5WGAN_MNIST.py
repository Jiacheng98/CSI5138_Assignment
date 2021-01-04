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
		)

	def forward(self, x):
				x = x.view(-1, 784)
				return self.main(x)



def main():


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



		optim_g = optim.RMSprop(generator.parameters(), lr=learning_rate)
		optim_d = optim.RMSprop(discriminator.parameters(), lr=learning_rate)




		losses_g = [] # to store generator loss after each epoch
		losses_d = [] # to store discriminator loss after each epoch
		emd_list = []
		images = [] # to store images generatd by the generator

		epoch_list = []



		noise = create_noise(sample_size, nz)


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

					em_distance += (-loss_d1)
					loss_d += loss_d1


				data_fake = generator(create_noise(b_size, nz))
				# train the generator
				loss_g1 = train_generator(generator, discriminator, optim_g, data_fake)
				loss_g += loss_g1
				
			generated_img = generator(noise).cpu().detach()
			generated_img = make_grid(generated_img, padding =2, normalize = True)
			save_generator_image(generated_img, f"./5IMAGE_WGAN_MNIST/epoch{epoch+1}.png")
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
		plt.title('WGAN_MNIST_EMD')
		plt.savefig(f"5WGAN_MNIST_emd_epoch.png")
		plt.close()


		fig = plt.figure()
		plt.plot(epoch_list, losses_g, color='red', label ='Generator loss' )
		plt.plot(epoch_list, losses_d, color='black', label = 'Critic loss')
		plt.legend(loc='best')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title('WGAN_MNIST_Loss')
		plt.savefig(f"5WGAN_MNIST_loss_epoch.png")
		plt.close()



def train_discriminator(discriminator, optimizer, data_real, data_fake):
	discriminator.train()
	b_size = data_real.size(0) # Get the batch size of data

	optimizer.zero_grad()
	critic_real = discriminator(data_real)
	critic_fake = discriminator(data_fake)
	loss_D = -(torch.mean(critic_real) - torch.mean(critic_fake))
	loss_D.backward()
	optimizer.step()

	for p in discriminator.parameters():
		p.data.clamp_(-weight_clip, weight_clip)

	return loss_D


def train_generator(generator, discriminator, optimizer, data_fake):
	generator.train()
	b_size = data_fake.size(0)

	optimizer.zero_grad()
	output = discriminator(data_fake)
	loss_G = -torch.mean(output)
	loss_G.backward()
	optimizer.step()

	return loss_G




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


		batch_size = 128
		epochs = 200
		sample_size = 64 # sampling a fixed size noise vector
		nz = 128 # latent vector size/noise vector size
		k = 5 # number of steps to apply to the discriminator
		weight_clip = 0.01
		learning_rate=5e-5



		main()