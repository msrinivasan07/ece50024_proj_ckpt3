import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt

#We define our own Euler ODE solver for testing the toy dataset
def euler(func, t, dt, y):
  return dt * func(t, y) #func is the derivative function

#Size of training dataset
data_sz = 1000

#generate toy dataset
true_y0 = torch.tensor([[2.,0.]]).cuda()
t = torch.linspace(0.,15.,data_sz).cuda() #timesteps at which the data points will be generated
true_A = torch.tensor([[-0.1,2.0],[-2.0,-0.1]]).cuda()

#function to generate spiral toy dataset
class data_gen(nn.Module):
  def forward(self, t, y):
    return torch.mm(y, true_A) #creates the derivative of the spiral function - matmul of y and A; this is used to generate the toy dataset

#While training actual NeuralODE, this function will be estimated by a neural network whose parameters we must learn by training
# Neural ODE model
class NeuralODE(nn.Module): #nn.Module keeps track of the entire nn framework in pytorch. It requires a forward function that determines the forward prop of the NN. It provides an autograd backward prop by itself
  def __init__(self, func):
    super().__init__()
    self.func = func #we need to pass the dynamics function

  def forward(self, y0, t, solver):
    solution = torch.empty(len(t), *y0.shape, dtype=y0.dtype, device=y0.device)
    solution[0] = y0 #sets the initial value for the IVP that will be solved by ODE solver

    j = 1
    for t0, t1 in zip(t[:-1], t[1:]):
      dy = solver(self.func, t0, t1 - t0, y0) #returns dy by which we must update y0
      y1 = y0 + dy #updates y(1) = y0 + dy
      solution[j] = y1 #stores the next y value in the solution array
      j += 1
      y0 = y1
    return solution

#First, we will forward propagate through the ODE model with our data_gen() function to generate the true trajectory of the toy spiral dataset
#This is done by executing the following code

with torch.no_grad():
  node = NeuralODE(func=data_gen()).cuda() #data_gen() is passed as the dynamics function
  true_y = node(y0=true_y0, t=t, solver=euler) #forward prop is performed to get true_y values at all timesteps with euler solver

#The dataset can be visualized using the following code
fig = plt.figure(figsize=(6, 6), facecolor='white')
ax = fig.add_subplot(111)
ax.set_title('Phase Portrait')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'green', label='true path') 
ax.scatter(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], color='red', label='samples', s=2)
ax.set_xlim(-2.5,2.5)
ax.set_ylim(-2.5,2.5)
plt.grid(True)
plt.legend()
plt.savefig('toy_dataset_trajectory') 
plt.show()
