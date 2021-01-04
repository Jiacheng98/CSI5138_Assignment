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

# print(torch.cuda.current_device())
# print(torch.cuda.device(0))
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())

random_seed = 2
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

def checkCuda():
    if torch.cuda.is_available():
        return True
    else:
        return False

n_epochs = 10
batch_size_train = 500
batch_size_test = 1000
learning_rate = 0.01
log_interval = 10


# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# # Normalize the test set same as training set without augmentation
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform = transforms.Compose([transforms.ToTensor()])

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



def getDataset():

    trainset = torchvision.datasets.CIFAR10('./data2', train=True, download=True, transform=transform) 
    cifar_trainset = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data2', train=False, download=True, transform=transform)
    cifar_testset = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2)

    return cifar_trainset, cifar_testset


train_dataset, test_dataset = getDataset()


# trains = enumerate(train_dataset)
# batch_idx, (trains_data, trains_targets) = next(trains)
# print("Train data batch: " + str(trains_data.shape)) 
#####Train data batch: torch.Size([500, 3, 32, 32])  
# print("Train target batch: " + str(trains_targets.shape))
##### batch: 500


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 32*32*3
output_dim = 10
model = LogisticRegressionModel(input_dim, output_dim)

gpu_available = checkCuda()
if gpu_available:
    device = torch.device("cuda:0")
    model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_dataset.dataset) for i in range(n_epochs + 1)]
test_accuracys = []
epoch_list = [i for i in range(0, n_epochs+1)]

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_dataset):
        if gpu_available:
            data = data.view(-1, 32*32*3).requires_grad_().to(device)
            target = target.to(device)
        else:
            data = data.view(-1, 32*32*3).requires_grad_()
        optimizer.zero_grad()
        output = model(data)
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



def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataset:
            if gpu_available:
                data = data.view(-1, 32*32*3).requires_grad_().to(device)
                target = target.to(device)
            else:
                data = data.view(-1, 32*32*3).requires_grad_()
            output = model(data)

            batch_total_loss = (criterion(output, target).item())*batch_size_test
            test_loss += batch_total_loss
            _, pred = torch.max(output.data, 1)
            if gpu_available:
                correct += (pred.cpu() == target.cpu()).sum()
            else:
                correct += (pred == target).sum()
    test_loss /= len(test_dataset.dataset)
    test_losses.append(test_loss)

    test_accuracy =  100. * correct / len(test_dataset.dataset)
    test_accuracys.append(test_accuracy/100.)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset.dataset),
        100. * correct / len(test_dataset.dataset)))





test()
for epoch in range(1, n_epochs+1):
  train(epoch)
  test()


fig = plt.figure()
plt.scatter(test_counter, test_losses, color='black', alpha = 1, zorder = 20)
plt.plot(train_counter, train_losses, color='pink', alpha = 1, zorder = 10)
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('cross-entropy loss')
plt.yticks(np.arange(0, 2.6, step=0.1)) 
plt.title('CIFAR10_Soft-max Regression')
plt.savefig('Softmax_CIFAR10')
plt.close()

fig1 = plt.figure()
plt.plot(epoch_list, test_accuracys, color='pink', alpha = 1)
plt.xlabel('epoch')
plt.ylabel('test data accuracy')
plt.xticks(range(min(epoch_list), max(epoch_list)+1, 1)) 
plt.yticks(np.arange(0, 1.1, step=0.1)) 
plt.title('CIFAR10_Soft-max Regression_Accuracy')
plt.savefig('Softmax_CIFAR10_accuracy')
plt.close()

