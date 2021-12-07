# only required to run python3 examples/cvt_arm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import math

import search
import utils

from mlp import evaluate_from_scratch
from mlp import evaluate_from_pretrained_weights
import multiprocessing

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def flatten_params(parameters):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**

    """
    l = [torch.flatten(p) for p in parameters]
    flat = torch.cat(l).view(-1,)
    return flat

def load_model_weights(model, weight_vec):
    """
    """
    sd = model.state_dict()
    curr_marker = 0

    for key, value in sd.items():
        curr_shape = value.shape
        tensor_len = np.prod(np.array(curr_shape))

        replace_with = weight_vec[curr_marker: curr_marker+tensor_len]
        replace_with = replace_with.reshape(curr_shape)

        sd[key] = torch.from_numpy(replace_with)

        curr_marker += tensor_len

    model.load_state_dict(sd)
    return model

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
total_parameters = count_parameters(model)
epochs = 30

px = utils.default_params.copy()

dim_x = total_parameters
params=utils.default_params
log_file=None
variation_operator=utils.variation

print("min {}".format(params['min']))
print("max {}".format(params['max']))

# setup the parallel processing pool
num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)

archive = {} # init archive (empty)
n_evals = 0 # number of evaluations since the beginning
b_evals = 0 # number evaluation since the last dump

robust_archive = {}
standard_archive = {}
total_iterations = 10
G = 5

z_arr = []

# main loop
for i in range(1, total_iterations+1):

    # random initialization
    if i < G:

        # for robustness
        initial_robust_weights = np.float32(np.random.uniform(low=-0.0357, high=0.0357, size=dim_x))
        model = load_model_weights(model, initial_robust_weights)
        model, testaccuracy = evaluate_from_scratch(model, epochs, mode="robust")
        flat_params = flatten_params(model.parameters())
        hashable_params = utils.make_hashable(flat_params)
        robust_archive[hashable_params] = testaccuracy

        # for standard
        initial_standard_weights = np.float32(np.random.uniform(low=-0.0357, high=0.0357, size=dim_x))
        model = load_model_weights(model, initial_standard_weights)
        model, testaccuracy = evaluate_from_scratch(model, epochs, mode="standard")
        flat_params = flatten_params(model.parameters())
        hashable_params = utils.make_hashable(flat_params)
        standard_archive[hashable_params] = testaccuracy

    else:  # variation/selection loop

        # randomly sample parents from robust and standard in order to mutate them
        # rand1 = np.random.randint(len(robust_archive), size=1)
        # rand2 = np.random.randint(len(standard_archive), size=1)
        rand1 = np.random.randint(len(robust_archive))
        rand2 = np.random.randint(len(standard_archive))
        
        # parent selection
        x, x_acc = list(robust_archive.items())[rand1]
        y, y_acc = list(standard_archive.items())[rand2]
        
        # copy & add variation
        z = variation_operator(np.array(x), np.array(y), params)
        z_arr.append(z)

        # evaluate z
        model = load_model_weights(model, z)

        robust_acc = evaluate_from_pretrained_weights(model, mode="robust")
        standard_acc = evaluate_from_pretrained_weights(model, mode="standard")

        current_robust_max = np.max(list(robust_archive.values()))
        current_standard_max = np.max(list(standard_archive.values()))

        hashable_z = utils.make_hashable(z)
        if robust_acc > current_robust_max:
            robust_archive[hashable_z] = robust_acc
        if standard_acc > current_standard_max:
            standard_archive[hashable_z] = standard_acc