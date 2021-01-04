import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import subprocess

# torch.set_grad_enabled(True)
# cuda = torch.device('cuda:0')
# print(torch.cuda.current_device())
# print(torch.cuda.device(0))
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())

random_seed = 43
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

def checkCuda():
    if torch.cuda.is_available():
        return True
    else:
        return False

n_epochs = 20
batch_size_train =500
batch_size_test = 1000
learning_rate = 1e-3
log_interval = 10
# momentum = 0.5
weight_decay = 0.0005


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def getDataset():

    trainset = torchvision.datasets.CIFAR10('./data2', train=True, download=True, transform=transform_train) 
    cifar_trainset = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data2', train=False, download=True, transform=transform_test)
    cifar_testset = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2)

    return cifar_trainset, cifar_testset


train_dataset, test_dataset = getDataset()

# trains = enumerate(train_dataset)
# batch_idx, (trains_data, trains_targets) = next(trains)
# print("Train data batch: " + str(trains_data.shape))
# print("Train target batch: " + str(trains_targets.shape))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define actual layers
      # C1 layer: input: 32*32*3, 32 kernels 3*3*3, output: 30*30*32
      # Dropout = 0.05

      # C2 layer: input: 30*30*32, 48 kernels 3*3*32, output: 28*28*48
      # Dropout = 0.05

      # C3 layer: input: 28*28*48, 64 kernels 3*3*32, output: 26*26*64
      # Dropout = 0.05

      # C4 layer: input: 26*26*64, 128 kernels 3*3*64, output: 24*24*128
      # Max-Pooling layer2: input: 12*12*128, kernel_size = 2, stride = 2, output: 12*12*128
      # Dropout = 0.05

      # C5 layer: input: 12*12*128, 256 kernels 3*3*128, output: 10*10*256
      # Max-Pooling layer2: input: 12*12*128, kernel_size = 2, stride = 2, output: 5*5*256
      # Dropout = 0.05

      # Falttern the tensors and pass through a MLP, reshape to 1 dimension with 5*5*256 = 6400 vectors
      # FC1 layer: input features: 6400, output features: 1024
      # FC2 layer: input features: 1024, output features: 512
      # FC3 layer: input features: 512, output features: 10
      # Output layer: Softmax layer

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.1),

            # Conv Layer block 2
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3),
            nn.BatchNorm2d(48),
            nn.Dropout2d(p=0.1),

            # Conv Layer block 3
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.1),

            # Conv Layer block 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),

            # Conv Layer block 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(6400, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
      # Define forward function, activation and max-pooling operations
        x = self.conv_layer(x)
        x = x.reshape(-1, 5*5*256)
        x = self.fc_layer(x)
        return x

network = Net()
gpu_available = checkCuda()
if gpu_available:
    device = torch.device("cuda:0")
    network.to(device)

optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay = weight_decay)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_dataset.dataset) for i in range(n_epochs + 1)]
test_accuracys = []
epoch_list = [i for i in range(0, n_epochs+1)]


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_dataset):
    if gpu_available:
        data = data.to(device)
        target = target.to(device)
    optimizer.zero_grad()
    output = network(data)
    # negative log-likelihodd loss between the output and the ground truth label
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_dataset.dataset),
        100. * batch_idx / len(train_dataset), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*batch_size_train) + ((epoch-1)*len(train_dataset.dataset)))
      # torch.save(network.state_dict(), '/results/model.pth')
      # torch.save(optimizer.state_dict(), '/results/optimizer.pth')


def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_dataset:
      if gpu_available:
        data = data.to(device)
        target = target.to(device)

      output = network(data)
      test_loss += (criterion(output, target).item()*batch_size_test)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_dataset.dataset)
  test_losses.append(test_loss)

  test_accuracy =  100. * correct / len(test_dataset.dataset)
  test_accuracys.append(test_accuracy/100.)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_dataset.dataset),
    100. * correct / len(test_dataset.dataset)))



test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()



fig = plt.figure()
plt.scatter(test_counter, test_losses, color='black', alpha = 1, zorder = 20)
plt.plot(train_counter, train_losses, color='pink', alpha = 1, zorder = 10)
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('cross entropy loss')
plt.ticklabel_format(style='plain')
plt.xticks(np.arange(min(test_counter), max(test_counter)+1 , 200000), fontsize = 6) 
plt.yticks(np.arange(0, 2.6, step=0.1)) 
plt.title('CIFAR10_Complex_CNN')
plt.savefig('Complex_CNN_CIFAR10')
plt.close()


fig1 = plt.figure()
plt.plot(epoch_list, test_accuracys, color='pink', alpha = 1)
plt.xlabel('epoch')
plt.ylabel('test data accuracy')
plt.xticks(range(min(epoch_list), max(epoch_list)+1, 1)) 
plt.yticks(np.arange(0, 1.1, step=0.1)) 
plt.title('CIFAR10_Complex_CNN_Accuracy')
plt.savefig('Complex_CNN_CIFAR10_Accuracy')
plt.close()


# test_examples = enumerate(test_dataset)
# batch_idx_examples, (data_examples, target_examples) = next(test_examples)
# # print("Test data batch: " + str(tests_data.shape))
# # print("Test target batch: " + str(tests_targets.shape))

# with torch.no_grad():
#   if gpu_available:
#     data_examples = data_examples.to(device)
#   output = network(data_examples)

# fig2 = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   data_examples = data_examples.cpu()
#   plt.imshow(data_examples[i][0], cmap='gray', interpolation='none')
#   plt.title("Prediction: {}".format(
#     output.data.max(1, keepdim=True)[1][i].item()))
#   plt.xticks([])
#   plt.yticks([])
# plt.savefig('Handwritten digits examples')
# plt.close()
