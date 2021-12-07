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
import itertools
import search
import utils

from mlp import evaluate_from_scratch
from mlp import evaluate_from_pretrained_weights

# from torch.multiprocessing import Pool, Process, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass

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

def evolutionary_strategy(hyperparams):
    iterations = 10000
    n = 100
    alpha, sigma = hyperparams

    print("current alpha is {}".format(alpha))
    print("current sigma is {}".format(sigma))

    model = MLP()
    total_parameters = count_parameters(model)

    # es step
    for t in range(iterations):
        flat_params = flatten_params(model.parameters())
        theta = flat_params.cpu().detach().numpy()
        param_len = len(theta)

        fitness_arr = []
        eps_arr = []

        for i in range(n):
            eps_i = np.random.normal(0, 1, size=param_len)
            eps_arr.append(eps_i)
            noise_i = (eps_i * sigma).astype(np.float32)
            perturbed_params = theta + noise_i

            # load model with perturbed params
            model = load_model_weights(model, perturbed_params)
            
            # evaluate fitness
            robust_acc = evaluate_from_pretrained_weights(model, mode="robust") / 100
            print("robust_acc {}".format(robust_acc*100))
            standard_acc = evaluate_from_pretrained_weights(model, mode="standard") / 100
            print("standard_acc {}".format(standard_acc*100))

            fitness_i = (robust_acc + standard_acc) / 2
            fitness_arr.append(fitness_i)

        fitness_arr = np.array(fitness_arr)
        eps_arr = np.array(eps_arr)
        summand = np.matmul(fitness_arr, eps_arr)

        # update quantity for parameters (theta)
        delta = (alpha / (n * sigma)) * summand

        # new theta
        theta += delta

    model = load_model_weights(model, theta)
    final_robust_acc = evaluate_from_pretrained_weights(model, mode="robust") / 100
    final_standard_acc = evaluate_from_pretrained_weights(model, mode="standard") / 100

    return model, final_robust_acc, final_standard_acc

# alpha_arr = [0.1, 0.01, 0.001, 0.0001]
# sigma_arr = [0.1, 0.5, 1, 1.5]
# hyperparam_arr = list(itertools.product(alpha_arr, sigma_arr))

# best hyperparams
# current alpha is 0.01
# current sigma is 0.1
hyperparam_arr = [(0.01, 0.1)]

for hp in hyperparam_arr:
	model, fra, fsa, = evolutionary_strategy(hp)
	print("final robust acc {}%, final standard acc {}% \n\n".format(fra*100, fsa*100))


# if __name__=="__main__":

#     # num_cores = torch.multiprocessing.cpu_count()
#     # pool = Pool(num_cores)
#     pool = Pool(10)
#     alpha_arr = [0.1, 0.01, 0.001, 0.0001]
#     sigma_arr = [0.1, 0.5, 1, 1.5]
#     hyperparam_arr = list(itertools.product(alpha_arr, sigma_arr))
#     s_list = pool.map(evolutionary_strategy, hyperparam_arr)