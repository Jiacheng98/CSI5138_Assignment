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
weight_decay = 0.0005


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


random_seed = 2
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


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
      # C1 layer: input: 32*32*3, 6 kernels 5*5*3, output: 28*28*6
      # Max-Pooling layer1: input: 28*28*6, kernel_size = 2, stride = 2, output: 14*14*6
      # C2 layer: input: 14*14*6, 16 kernels 5*5*6, output: 10*10*16
      # Max-Pooling layer2: input: 10*10*16, kernel_size = 2, stride = 2, output: 5*5*16
      # Falttern the tensors and pass through a MLP, reshape to 1 dimension with 5*5*16 = 400 vectors
      # FC1 layer: input features: 400, output features: 120
      # FC2 layer: input features: 120, output features: 84
      # FC3 layer: input features: 84, output features: 10
      # Output layer: Softmax layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # self.batchNorm1 = nn.BatchNorm2d(num_features=6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # self.batchNorm2 = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
      # Define forward function, activation and max-pooling operations
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = self.dropout(x)
        x = F.relu(F.max_pool2d(self.conv2(x),  kernel_size=2, stride=2))
        x = self.dropout(x)
        x = x.reshape(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
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
plt.title('CIFAR10_Simple_CNN')
plt.savefig('Simple_CNN_CIFAR10')
plt.close()


fig1 = plt.figure()
plt.plot(epoch_list, test_accuracys, color='pink', alpha = 1)
plt.xlabel('epoch')
plt.ylabel('test data accuracy')
plt.xticks(range(min(epoch_list), max(epoch_list)+1, 1)) 
plt.yticks(np.arange(0, 1.1, step=0.1)) 
plt.title('CIFAR10_Simple_CNN_Accuracy')
plt.savefig('Simple_CNN_CIFAR10_accuracy')
plt.close()

