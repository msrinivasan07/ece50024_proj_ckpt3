import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt

data_sz = 1000

#generate toy dataset
true_y0 = torch.tensor([[2.,0.]]).cuda()
t = torch.linspace(0.,15.,data_sz).cuda()
true_A = torch.tensor([[-0.1,2.0],[-2.0,-0.1]]).cuda()