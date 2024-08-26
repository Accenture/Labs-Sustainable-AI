from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
# from tqdm import tqdm # for progress bar

# import sys
# sys.path.append('../')
# from save_data import save_data
# import time

# Decide if we deplay info during training:
# display_train = False

#####################
##### Section 1 #####
#####################

# Definition of the ML model, training epoch, and testing function

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_0(args, model, device, train_loader, optimizer, epoch):
    """ Training without display of loss """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def train_1(args, model, device, train_loader, optimizer, epoch):
    """ Training with display of loss """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#######################
##### Section 2.1 #####
#######################

# Creating the Params class that will store all the parameters of the experiment

# class Params():
#     def __init__(self, batch_size = 64, test_batch_size = 1000, epochs = 14, lr = 1.0, 
#                 gamma = 0.7, no_cuda = False, no_mps = False, dry_run = False, seed = 1, 
#                 log_interval = 10, save_model = False, data_folder = './data', 
#                 nb_batch_inferences = 100, dev_test = False):
        # self.batch_size = batch_size            # input batch size for training (default: 64)
        # self.test_batch_size = test_batch_size  # input batch size for testing (default: 1000)
        # self.epochs = epochs                    # number of epochs to train (default: 14)
        # self.lr = lr                            # learning rate (default: 1.0)
        # self.gamma = gamma                      # learning rate step gamma (default: 0.7)
        # self.no_cuda = no_cuda                  # disables CUDA training
        # self.no_mps = no_mps                    # disables macOS GPU training
        # self.dry_run = dry_run                  # quickly check a single pass
        # self.seed = seed                        # random seed (default: 1)
        # self.log_interval = log_interval        # how many batches to wait before logging training status (default: 10)
        # self.save_model = save_model            # For Saving the current Model
        # self.data_folder = data_folder
        # self.nb_batch_inferences = nb_batch_inferences

        # if dev_test:
        #     # For tests:
        #     self.epochs = 2
        #     self.nb_batch_inferences = 10

def create_dataloaders(use_cuda, args):
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,'pin_memory': True,'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST(args.data_folder, train=True, download=True,
                    transform=transform)
    dataset2 = datasets.MNIST(args.data_folder, train=False,
                    transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return(train_loader, test_loader)