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

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

def checkCuda():
    if torch.cuda.is_available():
        return True
    else:
        return False

n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 1e-3
log_interval = 10


def getDataset():
  mnist_trainset = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,  transform=transforms.Compose([
                                               transforms.ToTensor()])), batch_size=batch_size_train, shuffle=True)
  mnist_testset = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                                              transforms.ToTensor()])), batch_size=batch_size_test, shuffle=False)

  return mnist_trainset, mnist_testset


train_dataset, test_dataset = getDataset()

# trains = enumerate(train_dataset)
# batch_idx, (trains_data, trains_targets) = next(trains)
# print("Train data batch: " + str(trains_data.shape))
# print("Train target batch: " + str(trains_targets.shape))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define actual layers
      # C1 layer: input: 28*28*1, 6 kernels 5*5, output: 24*24*6
      # Max-Pooling layer1: input: 24*24*6, kernel_size = 2, stride = 2, output: 12*12*6
      # C2 layer: input: 12*12*6, 12 kernels 5*5, output: 8*8*12
      # Max-Pooling layer2: input: 8*8*12, kernel_size = 2, stride = 2, output: 4*4*12
      # Falttern the tensors and pass through a MLP, reshape to 1 dimension with 4*4*12 = 192 vectors
      # FC1 layer: input features: 192, output features: 60
      # FC2 layer: input features: 60, output features: 10
      # Output layer: Softmax layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
      # Define forward function, activation and max-pooling operations
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x),  kernel_size=2, stride=2))
        x = x.reshape(-1, 12*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

network = Net()
gpu_available = checkCuda()
if gpu_available:
    device = torch.device("cuda:0")
    network.to(device)

optimizer = optim.Adam(network.parameters(), lr=learning_rate)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_dataset.dataset) for i in range(n_epochs + 1)]
test_accuracys=[]
epoch_list = [i for i in range(0, n_epochs+1)]


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_dataset):
    if gpu_available:
        data = data.to(device)
        target = target.to(device)
    optimizer.zero_grad()
    output = network(data)
    # cross-entropy loss between the output and the ground truth label
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_dataset.dataset),
        100. * batch_idx / len(train_dataset), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_dataset.dataset)))
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
      test_loss += F.cross_entropy(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_dataset.dataset)
  test_losses.append(test_loss)


  test_accuracy = 100. * correct / len(test_dataset.dataset)
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
plt.yticks(np.arange(0, 2.6, step=0.1)) 
plt.title('MNIST_CNN')
plt.savefig('CNN_MNIST')
plt.close()



fig1 = plt.figure()
plt.plot(epoch_list, test_accuracys, color='pink', alpha = 1)
plt.xlabel('epoch')
plt.ylabel('test data accuracy')
plt.xticks(range(min(epoch_list), max(epoch_list)+1, 1)) 
plt.yticks(np.arange(0, 1.1, step=0.1)) 
plt.title('MNIST_CNN_Accuracy')
plt.savefig('CNN_MNIST_accuracy')
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
