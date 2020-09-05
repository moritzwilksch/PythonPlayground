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

xtrain = torch.Tensor(xtrain.values).cuda()
ytrain = torch.Tensor(ytrain.values).cuda()
xval = torch.Tensor(xval.values).cuda()
yval = torch.Tensor(yval.values).cuda()

trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtrain, ytrain), batch_size=32, shuffle=True, num_workers=8)

#%%
net = nn.Sequential(
    nn.Linear(12, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    # nn.ReLU(),
    # nn.Linear(10, 1)
)

net = net.cuda()

opt = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.L1Loss()

#%%
train_losses = []
val_losses = []

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