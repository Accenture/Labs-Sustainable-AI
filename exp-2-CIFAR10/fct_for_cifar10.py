import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tqdm import tqdm # for progress bar
# import os
# import sys
# sys.path.append('../')
# from save_data import save_data
# import pandas as pd

#####################
##### Section 1 #####
#####################

# Definition of the ML model, and training epoch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_0(net, device, trainloader, criterion, optimizer, epoch):
    """ Training with display of loss """
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def train_1(net, device, trainloader, criterion, optimizer, epoch):
    """ Training with display of loss """
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# class Params():
#     def __init__(self, batch_size = 4, lr = 0.001, momentum = 0.9,
#                 seed = 1, data_folder = './data'):
#         self.batch_size = batch_size  # input batch size for training (default: 4)
#         self.test_batch_size = batch_size  # input batch size for inference (default: 4)
#         self.lr = lr                  # learning rate (default: 0.001)
#         self.momentum = momentum
#         self.seed = seed              # random seed (default: 1)
#         self.data_folder = data_folder

def create_dataloaders(args):

    # --- Note from pytorch --- #
    # If running on Windows and you get a BrokenPipeError, 
    # try setting the num_worker of torch.utils.data.DataLoader() to 0
    # ---                   --- #

    # load CIFRAR10 using torchvision + normalize it
    # -> PILImage images of range [0, 1]
    # + transform these images to Tensors of normailzed range [-1, 1]

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=args.data_folder, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data_folder, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)
    return(trainloader, testloader)






def main():

    no_accelerator = True

    # Setting the parameters and eventual accelerator

    # Settings
    batch_size = 4           # input batch size for training (default: 64)
    epochs = 10              # number of epochs to train (default: 14)
    lr = 0.001               # learning rate (default: 1.0)
    momentum = 0.9           # ...

    use_cuda = not no_accelerator and torch.cuda.is_available()
    use_mps = not no_accelerator and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device is: ", device)

    #####################
    ##### Section 3 #####
    #####################

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)   

    if display_images == True:
        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # Instance of the ML model
    net = Net().to(device)

    # Define a Loss function and optimizer:
    # Classification Cross-Entropy loss and SGD with momentum
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    #####################
    ##### Section 4 #####
    #####################

    # Training

    # -- --- --- -- #
    # -  CC block - #
    # -           - #
    
    # We create the tracker's instance:
    output_file = "output_cc.csv"
    
    tracker = EmissionsTracker(output_file = output_file)

    print("# ---------------------- #")
    print("# --- training start --- #")
    print("# ---------------------- #")
    tracker.start()
    # -           - #
    # -           - #
    # -- --- --- -- #

    if display_train == False:
        train = train_0
        # Loop over the data iterator, feed the inputs to the network and optimize.
        for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
            train(net, device, trainloader, criterion, optimizer, epoch)
    else:
        train = train_1
        # Loop over the data iterator, feed the inputs to the network and optimize.
        for epoch in range(epochs):  # loop over the dataset multiple times
            train(net, device, trainloader, criterion, optimizer, epoch)

    # -- --- --- -- #
    # -  CC block - #
    # -           - #
    print("# --------------------- #")
    print("# --- training stop --- #")
    print("# --------------------- #")
    tracker.stop()
    # -           - #
    # -           - #
    # -- --- --- -- #

    # save the trained model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    # more on saving Pytorch models: https://pytorch.org/docs/stable/notes/serialization.html

    #####################
    ##### Section 5 #####
    #####################

    # Inference

    # recover the saved model
    PATH = "./cifar_net.pth"
    net = Net().to(device)
    net.load_state_dict(torch.load(PATH)) 

    nb_batch_inferences = 100
    nb_inferences = nb_batch_inferences*batch_size

    # -- --- --- -- #
    # -  CC block - #
    # -           - #
    print("# ----------------------- #")
    print("# --- inference start --- #")
    print("# ----------------------- #")
    output_file = "output_cc.csv"
    # measure_power_secs = 0.1

    # save_to_logger
    tracker = EmissionsTracker(output_file = output_file)
    tracker.start()
    # -           - #
    # -           - #
    # -- --- --- -- #      

    for kk in  tqdm(range(nb_batch_inferences)):
        inputs, targets = next(iter(testloader))
        inputs = inputs.to(device)
        outputs = net(inputs)
        _, one_pred = torch.max(outputs, 1)


    # -- --- --- -- #
    # -  CC block - #
    # -           - #    
    print("# -------------------------- #")
    print("# --- tag inference stop --- #")
    print("# -------------------------- #")
    tracker.stop()
    # -           - #
    # -           - #
    # -- --- --- -- #



    # -- --- --- -- #
    # -  CC block - #
    # -           - #

    # Saving the data in the json file

    experience = "cifar10"
    device_type = device.type
    calculator = "CC"

    # -------------------------------- #
    # - CHANGE DEPENDING ON COMPUTER - #
    #                                  #
    computer = "linux_alienware"
    #                                  #
    # -------------------------------- #


    file = pd.read_csv("output_cc.csv")

    df=pd.DataFrame(file)

    # For training:
    ml_phase = "training"
    meas_epochs = epochs
    meas_time = df["duration"].iloc[-2]
    meas_energy = df["energy_consumed"].iloc[-2]
    meas_co2 = df["emissions"].iloc[-2]
    save_data(experience, ml_phase, computer, device_type, calculator, meas_epochs, meas_time, meas_energy, meas_co2)

    # For inference:
    ml_phase = "inference"
    meas_epochs = "N/A"
    meas_time = df["duration"].iloc[-1]
    meas_energy = df["energy_consumed"].iloc[-1]
    meas_co2 = df["emissions"].iloc[-1]
    meas_time = meas_time/nb_inferences
    meas_energy = meas_energy/nb_inferences
    meas_co2 = meas_co2/nb_inferences

    save_data(experience, ml_phase, computer, device_type, calculator, meas_epochs, meas_time, meas_energy, meas_co2)
   
    os.remove("output_cc.csv")
    # -           - #
    # -           - #
    # -- --- --- -- #

if __name__ == '__main__':
    main()