import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np

# mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
# mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

bs = 64 # batch size in every epoch

train_dataset = datasets.MNIST(root = 'data/', train=True, download=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

test_dataset =  datasets.MNIST(root = 'data/', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

# trainning set
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=bs, 
                                           shuffle=True)

# test set
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=100, 
                                          shuffle=True)

indices = np.load("/home/shadabs3/532J_project/CVTME_RobustNN/invariance_examples/indexs.npy")
l0_inv_examples = np.load("/home/shadabs3/532J_project/CVTME_RobustNN/invariance_examples/l0/automated.npy")
original_labels = test_dataset.targets[indices].numpy()
human_labels = np.load("/home/shadabs3/532J_project/CVTME_RobustNN/invariance_examples/l0/automated_labels.npy")

successful_indices = np.where(original_labels != human_labels)[0]

np.save("/home/shadabs3/532J_project/CVTME_RobustNN/invariance_examples/final_l0/inv_adv_examples.npy", l0_inv_examples[successful_indices])
np.save("/home/shadabs3/532J_project/CVTME_RobustNN/invariance_examples/final_l0/original_labels.npy", original_labels[successful_indices])
np.save("/home/shadabs3/532J_project/CVTME_RobustNN/invariance_examples/final_l0/human_labels.npy", human_labels[successful_indices])
