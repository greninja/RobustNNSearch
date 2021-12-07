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

def count_parameters(model):    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
total_iterations = 100
G = 50

save_path = os.path.join("saved_stuff", "G_"+str(G))
if not os.path.exists(save_path):
    os.makedirs(save_path)

px = utils.default_params.copy()

robust_archive, standard_archive = search.compute(
			                     	   total_parameters,
                                       model,
                                       epochs,
                                       G,
                                       total_iterations,
                                       "iso_dd",
                                       save_path,
			                     	   params=px
			                        )

# what to do with 'archive'
max_robust_nn = np.array(max(robust_archive, key=robust_archive.get))
max_standard_nn = np.array(max(standard_archive, key=standard_archive.get))

torch.save(max_robust_nn, os.path.join(save_path, "max_robust_nn.pt"))
torch.save(max_standard_nn, os.path.join(save_path, "max_standard_nn.pt"))

# crossover the final ones using "iso_dd"
crossover_nn = utils.variation(max_robust_nn, max_standard_nn, "iso_dd", px)
model = search.load_model_weights(model, crossover_nn)
robust_acc = evaluate_from_pretrained_weights(model, mode="robust")
standard_acc = evaluate_from_pretrained_weights(model, mode="standard")
print("ISO_DD crossover results: Final robust acc {}, Final standard acc {}".format(robust_acc, standard_acc))

# crossover the final ones using "sbx"
crossover_nn = utils.variation(max_robust_nn, max_standard_nn, "sbx", px)
model = search.load_model_weights(model, crossover_nn)
robust_acc = evaluate_from_pretrained_weights(model, mode="robust")
standard_acc = evaluate_from_pretrained_weights(model, mode="standard")
print("SBX crossover results: Final robust acc {}, Final standard acc {}".format(robust_acc, standard_acc))

# take linear combination of the final ones
alpha = 0.5
lc_nn = (alpha * max_robust_nn) + ((1-alpha) * max_standard_nn)
model = search.load_model_weights(model, lc_nn)
robust_acc = evaluate_from_pretrained_weights(model, mode="robust")
standard_acc = evaluate_from_pretrained_weights(model, mode="standard")
print("Linear Combination (alpha={}) results: Final robust acc {}, Final standard acc {}".format(alpha, robust_acc, standard_acc))