#%%
import torch
import torch.nn as nn
from torch import optim
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#%%
from sklearn.model_selection import train_test_split
df = pd.get_dummies(sns.load_dataset('tips'))
xtrain, xval, ytrain, yval = train_test_split(df.drop('tip', axis=1), df['tip'])
xtrain.info()

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
xtrain = ss.fit_transform(xtrain)
xval = ss.transform(xval)

xtrain = torch.Tensor(xtrain)# .cuda()
ytrain = torch.Tensor(ytrain.values)# .cuda()
xval = torch.Tensor(xval)# .cuda()
yval = torch.Tensor(yval.values)# .cuda()

BATCHSIZE = 4
trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtrain, ytrain), batch_size=BATCHSIZE, shuffle=True)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


#%%
net = nn.Sequential(
    nn.Linear(12, 20),
    nn.ReLU(),
    nn.Linear(20, 1),
    #nn.ReLU(),
    #nn.Linear(5, 1)
)

net.apply(init_weights)

# net = net# .cuda()
opt = optim.Adam(net.parameters(), lr=10**-2.8)
criterion = nn.L1Loss()

from torch_lr_finder import LRFinder
lrf = LRFinder(net, opt, criterion)
lrf.range_test(train_loader=trainloader, start_lr=0.0001, end_lr=1)
lrf.plot()
lrf.reset()

#%%
train_losses = []
val_losses = []
scheduler = torch.optim.lr_scheduler.CyclicLR(opt, 10**-3, 10**-2, mode='exp_range', step_size_up=(xtrain.size(0)/BATCHSIZE)*8, cycle_momentum=False)

for epoch in range(100):
  for batch in trainloader:
    inputs, labels = batch
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = net(inputs)

    # Zero gradients, perform a backward pass, and update the weights.
    opt.zero_grad()

    # perform a backward pass (backpropagation)
    loss = criterion(y_pred, labels.view(-1, 1))
    loss.backward()

    # Update the parameters
    opt.step()
    scheduler.step()
  # Compute and print loss
  
  with torch.no_grad():
    # net.eval()
    train_loss = criterion(net(xtrain), ytrain.view(-1, 1))
    val_loss = criterion(net(xval), yval.view(-1, 1))

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if epoch % 10 == 0:
      print('epoch: ', epoch,' loss: ', train_loss , 'val-loss', val_loss)

#%%
plt.plot(train_losses)
plt.plot(val_losses)

from sklearn.metrics import mean_absolute_error
print(f"MAE = {mean_absolute_error(yval, net(xval).detach())}")