import os 
import math 
import torch
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('seaborn-whitegrid')

vanilla_robust_max = np.max(torch.load("saved_stuff/eval_mlp/robust_acc_arr.pt"))
G_10_robust_max = np.max(torch.load("saved_stuff/G_10/robust_acc_arr.pt"))
G_20_robust_max = np.max(torch.load("saved_stuff/G_20/robust_acc_arr.pt"))
G_30_robust_max = np.max(torch.load("saved_stuff/G_30/robust_acc_arr.pt"))
G_40_robust_max = np.max(torch.load("saved_stuff/G_40/robust_acc_arr.pt"))
G_50_robust_max = np.max(torch.load("saved_stuff/G_50/robust_acc_arr.pt"))

vanilla_standard_arr = torch.load("saved_stuff/eval_mlp/standard_acc_arr.pt")

our_robust_arr = torch.load("saved_stuff/G_50/robust_acc_arr.pt")
our_standard_arr = torch.load("saved_stuff/G_50/standard_acc_arr.pt")

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

# Multiple lines in same plot
x = np.arange(1, 101)

# Robustness Plot
plt.plot(x, vanilla_robust_arr, label="vanilla")
plt.plot(x, our_robust_arr, label="our approach")
plt.legend()

# Decorate
plt.xlabel('# Iterations')
plt.ylabel('Robustness (%)')
plt.title("Vanilla vs Our approach (G=50)")
# plt.xlim(1,10)
# plt.ylim(-1.0 , 2.5)
plt.savefig("robustness.png")
plt.show()

# # Multiple lines in same plot
# x = np.arange(1, 101)

# # Robustness Plot
# plt.plot(x, vanilla_standard_arr, label="vanilla")
# plt.plot(x, our_standard_arr, label="our approach")
# plt.legend()

# # Decorate
# plt.xlabel('# Iterations')
# plt.ylabel('Accuracy')
# plt.title("Standard accuracy")
# # plt.xlim(1,10)
# # plt.ylim(-1.0 , 2.5)
# plt.show()