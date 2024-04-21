import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

#Reading the data
data = pd.read_csv('/kaggle/input/covid-jhu/time_series_covid19_confirmed_global.csv')
data.head()

#In this example, we consider the US data
us_data = data.loc[data['Country/Region']=='US']
dates = us_data.columns[4:]

t = np.arange(len(dates))
cases = us_data.iloc[:,4:].values
cases = cases.T

#Rescaling the data
from sklearn.preprocessing import StandardScaler
sc_t = StandardScaler()
sc_y = StandardScaler()

t = sc_t.fit_transform(t.reshape(-1,1))
cases = sc_y.fit_transform(cases)

#We define our own Euler ODE solver
def euler(func, t, dt, y):
  return dt * func(t, y) #func is the derivative function

#Converting data to PyTorch tensors
t = torch.tensor(t,dtype=torch.float).cuda()
true_y = torch.tensor(cases,dtype=torch.float).cuda()

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

class ODEFunc(nn.Module):
  def __init__(self): #this is the implementation of actual ODE where forward function is a neural network
    super().__init__()
    #The input layer of the NN has 2 nodes because the toy dataset is 2-dimensional
      # We create a hidden layer with 50 nodes and use tanh() activation function
      #The output layer must have 2 nodes as well because we are trying to output the value at the next timestep (it will be 2 dimensional as well)
    self.net = nn.Sequential(nn.Linear(1, 500),
                             nn.ReLU(),
                             nn.Linear(500, 1)) #architecture of the neural network
    for m in self.net.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1) #random initilisation of NN weights from a normal distribution
        nn.init.constant_(m.bias, val=0) #initialises bias for hidden layer with 0 value

  def forward(self, t, y):
    output = self.net(y) #forward propagates the input y through the nn
    return output
  
  def forward_with_grad(self, t, y, grad_outputs):
    output = self.forward(t, y)
    adj = grad_outputs

    adfdz, adfdt, *adfdp = torch.autograd.grad((output,),(t,y) + tuple(self.parameters()),grad_outputs=(adj),allow_unused=True,retain_graph=True)

    return output, adfdz, adfdt, adfdp

#Training the model
niters = 400
node = NeuralODE(func=ODEFunc()).cuda()
optimizer = optim.Adam(node.parameters(), lr=1e-3)
loss_history = []
for iter in tqdm(range(niters + 1)): #tqdm is simply used to create a progress bar for the iteration run
  optimizer.zero_grad()
  pred_y = node(y0=true_y[0], t=t, solver=euler) #uses euler solver to forward propagate the batch through the node object
  loss = torch.mean(torch.square(pred_y - true_y)) #calculates the L2-norm loss
  loss.backward() #backpropagates through the loss
  optimizer.step() #updates the parameters of the defined NN in the NeuralODE class

  if iter % 50 == 0:
    with torch.no_grad(): #gradients are not calculated as this is used to simply visualize the status of training
      pred_y = node(true_y[0], t, solver=euler)
      loss = torch.mean(torch.square(pred_y - true_y))
      loss_history.append(loss.item())
      print('Iter {:04d} | Total Loss {:.6f}'.format(iter, loss.item()))

#Visualizing the trained model
plt.plot(sc_t.inverse_transform(t.cpu().numpy()),sc_y.inverse_transform(true_y.cpu().numpy()),label='True')
plt.plot(sc_t.inverse_transform(t.cpu().numpy()),sc_y.inverse_transform(pred_y.cpu().numpy()),label='Predicted')
plt.xlabel("Days")
plt.ylabel("Number of Cases")
plt.legend()
plt.show()

#Visualizing the loss
plt.plot([0,50,100,150,200,250,300,350,400],loss_history)
plt.xlabel("Iterations")
plt.ylabel("L2 Loss")
plt.show()
