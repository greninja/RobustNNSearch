# only required to run python3 examples/cvt_arm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import os
import math
import numpy as np
import multiprocessing

import utils
from mlp import evaluate_from_scratch
from mlp import evaluate_from_pretrained_weights

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

def compute(dim_x,
            model,
            epochs,
            G,
            total_iterations,
            variation_type,
            save_path,
            params=utils.default_params):

    """
    """
    
    # setup the parallel processing pool
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    archive = {} # init archive (empty)

    robust_archive = {}
    standard_archive = {}

    print("G is {}".format(G))
    print("total_iterations is {}".format(total_iterations))    

    robust_acc_arr = []
    standard_acc_arr = []

    # main loop
    for i in range(1, total_iterations+1):

        # random initialization
        if i < G:

            # for robustness
            initial_robust_weights = np.float32(np.random.uniform(low=-0.0357, high=0.0357, size=dim_x))
            model = load_model_weights(model, initial_robust_weights)
            model, robust_acc = evaluate_from_scratch(model, epochs, mode="robust")
            flat_params = flatten_params(model.parameters())
            hashable_params = utils.make_hashable(flat_params)
            robust_archive[hashable_params] = robust_acc
            robust_acc_arr.append(robust_acc)

            # for standard
            initial_standard_weights = np.float32(np.random.uniform(low=-0.0357, high=0.0357, size=dim_x))
            model = load_model_weights(model, initial_standard_weights)
            model, standard_acc = evaluate_from_scratch(model, epochs, mode="standard")
            flat_params = flatten_params(model.parameters())
            hashable_params = utils.make_hashable(flat_params)
            standard_archive[hashable_params] = standard_acc
            standard_acc_arr.append(standard_acc)

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
            z = utils.variation(np.array(x), np.array(y), variation_type, params)

            # evaluate z
            model = load_model_weights(model, z)

            robust_acc = evaluate_from_pretrained_weights(model, mode="robust")
            robust_acc_arr.append(robust_acc)
            standard_acc = evaluate_from_pretrained_weights(model, mode="standard")
            standard_acc_arr.append(standard_acc)

            current_robust_max = np.max(list(robust_archive.values()))
            current_standard_max = np.max(list(standard_archive.values()))

            hashable_z = utils.make_hashable(z)
            if robust_acc > current_robust_max:
                robust_archive[hashable_z] = robust_acc
            if standard_acc > current_standard_max:
                standard_archive[hashable_z] = standard_acc

    torch.save(robust_acc_arr, os.path.join(save_path, "robust_acc_arr.pt"))
    torch.save(standard_acc_arr, os.path.join(save_path, "standard_acc_arr.pt"))

    return robust_archive, standard_archive