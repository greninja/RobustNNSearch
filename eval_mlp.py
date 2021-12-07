import time
# only required to run python3 examples/cvt_arm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import os 
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
batch_size = 64 # batch size in every epoch

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data 
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        img, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        
train_dataset = datasets.MNIST(root = 'data/', train=True, download=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

# training set
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=4,
                                           drop_last=True)


test_dataset =  datasets.MNIST(root = 'data/', train=False, download=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))


# test set
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=100, 
                                          shuffle=True,
                                          num_workers=4,
                                          drop_last=False)

inv_adv_examples = np.load("invariance_examples/final_l0/inv_adv_examples.npy") # visualize this for sanity check
human_labels = np.load("invariance_examples/final_l0/human_labels.npy")

inv_eg_dataset = CustomDataset(data=inv_adv_examples,
                               targets=human_labels,
                               transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

inv_eg_dataloader = torch.utils.data.DataLoader(inv_eg_dataset,
                                                batch_size=10,
                                                shuffle=True,
                                                num_workers=4,
                                                drop_last=False)

class MLP(nn.Module):
    def __init__(self, input_size=784, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, output_size)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

model = MLP()
model = model.to(device)

epochs = 100

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
model.train() # prep model for training

standard_acc_arr = []
robust_acc_arr = []

for epoch in range(epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)

    #--------------------------------------------------------------------------------------
    # initialize lists to monitor test loss and accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval() # prep model for *evaluation*

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
    standard_acc_arr.append(test_accuracy)
    print('\n Standard test accuracy: %2d%% (%2d/%2d)' % (test_accuracy, 
                                                          np.sum(class_correct), 
                                                          np.sum(class_total)))
    #--------------------------------------------------------------------------------------        

    #--------------------------------------------------------------------------------------
    # initialize lists to monitor test loss and accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval() # prep model for *evaluation*

    for data, target in inv_eg_dataloader:
        data, target = data.to(device), target.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
    robust_acc_arr.append(test_accuracy)
    print('\n Robust test accuracy: %2d%% (%2d/%2d)' % (test_accuracy,
                                                        np.sum(class_correct), 
                                                        np.sum(class_total)))
    #--------------------------------------------------------------------------------------

save_path = "saved_stuff/eval_mlp/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

torch.save(standard_acc_arr, os.path.join(save_path, "standard_acc_arr.pt"))
torch.save(robust_acc_arr, os.path.join(save_path, "robust_acc_arr.pt"))
