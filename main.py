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
ax.set_title('Toy Dataset')
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

#We can also define a visualize function using this code to be reused later
def visualize(true_y, pred_y=None):
  fig = plt.figure(figsize=(6, 6), facecolor='white')
  ax = fig.add_subplot(111)
  ax.set_title('Toy Dataset')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'green', label='true path')
  ax.scatter(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], color='blue', label='samples', s=2)
  if pred_y is not None:
    ax.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'red', label='learned path')
  ax.set_xlim(-2.5, 2.5)
  ax.set_ylim(-2.5, 2.5)
  plt.legend()
  plt.grid(True)
  plt.show()

#Now, we need to define the Neural ODE that we will use to train our model
#In this case, we do not know the true dynamics of the state changes
#So, we must represent the dynamics using a Neural Network
# We aim to train this Neural Network using samples from our toy dataset and update its parameters. 
class ODEFunc(nn.Module):
  def __init__(self): #this is the implementation of actual ODE where forward function is a neural network
    super().__init__()
    #The input layer of the NN has 2 nodes because the toy dataset is 2-dimensional
      # We create a hidden layer with 50 nodes and use tanh() activation function
      #The output layer must have 2 nodes as well because we are trying to output the value at the next timestep (it will be 2 dimensional as well)
    self.net = nn.Sequential(nn.Linear(2, 50),
                             nn.Tanh(),
                             nn.Linear(50, 2)) #architecture of the neural network
    for m in self.net.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1) #random initilisation of NN weights from a normal distribution
        nn.init.constant_(m.bias, val=0) #initialises bias for hidden layer with 0 value

  def forward(self, t, y):
    output = self.net(y) #forward propagates the input y through the nn
    return output

#Now, we extract batches from the generated toy spiral dataset to train the Neural ODE
batch_time = 10
batch_size = 16

def get_batch():
  s = torch.from_numpy(np.random.choice(np.arange(data_sz - batch_time, dtype=np.int64), batch_size, replace=False))
  batch_y0 = true_y[s]  
  batch_t = t[:batch_time] 
  batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0) 
  return batch_y0.cuda(), batch_t.cuda(), batch_y.cuda()

## Train
niters = 400 #number of training epochs

node = NeuralODE(func=ODEFunc()).cuda() #creates an object of the class NeuralODE with ODEFunc as the dynamics function
optimizer = optim.RMSprop(node.parameters(), lr=1e-3) #RMSprop is used as the optimizer for training

start_time = time.time() #to measure computation time

for iter in tqdm(range(niters + 1)): #tqdm is simply used to create a progress bar for the iteration run
  optimizer.zero_grad()
  batch_y0, batch_t, batch_y = get_batch() #extracts a batch of data from the true_y values created
  pred_y = node(y0=batch_y0, t=batch_t, solver=euler) #uses euler solver to forward propagate the batch through the node object
  loss = torch.mean(torch.square(pred_y - batch_y)) #calculates the L2-norm loss
  loss.backward() #backpropagates through the loss
  optimizer.step() #updates the parameters of the defined NN in the NeuralODE class

  if iter % 50 == 0:
    with torch.no_grad(): #gradients are not calculated as this is used to simply visualize the status of training
      pred_y = node(true_y0, t, solver=euler)
      loss = torch.mean(torch.abs(pred_y - true_y))
      print('Iter {:04d} | Total Loss {:.6f}'.format(iter, loss.item()))
      visualize(true_y, pred_y)

end_time = time.time() - start_time #calculates starting time - ending time
print('process time: {} sec'.format(end_time))
