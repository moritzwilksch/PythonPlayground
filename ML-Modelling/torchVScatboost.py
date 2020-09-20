#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import numpy as np

columns = ['Age','Workclass','fnlgwt','Education','Education Num','Marital Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours/Week','Country','<=50K']

training = pd.read_csv('adult-training.csv', sep=', ', names=columns)
# training[training == '?'] = np.nan
# testing[testing == '?'] = np.nan
testing = pd.read_csv('adult-test.csv', sep=', ', names=columns)

# Convert to bool
training['<=50K'] = (training['<=50K'] == '<=50K')
testing['<=50K'] = (testing['<=50K'] == '<=50K.')

# Convert object to category
training.loc[:, training.dtypes == 'object'] = training.loc[:, training.dtypes == 'object'].astype('category')
testing.loc[:, testing.dtypes == 'object'] = testing.loc[:, testing.dtypes == 'object'].astype('category')
catcols = training.drop('<=50K', axis=1).select_dtypes('category').columns.tolist()

#%%
training[training == '?'] = np.nan
testing[testing == '?'] = np.nan

for col in catcols:
    training[col] = training[col].cat.add_categories('NA')
    testing[col] = testing[col].cat.add_categories('NA')
    training[col] = training[col].fillna('NA')
    testing[col] = testing[col].fillna('NA')




xtest = testing.drop('<=50K', axis=1)
ytest = testing['<=50K']
#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

xtrain, xval, ytrain, yval = train_test_split(training.drop('<=50K', axis=1), training['<=50K'], random_state=1234)

#%%

cbc = CatBoostClassifier(cat_features=catcols, eval_metric='Accuracy')
cbc.fit(xtrain, ytrain, verbose=50, eval_set=(xval, yval), early_stopping_rounds=100)

#%%
def print_eval(ytrue, preds):
    print("\n" + classification_report(ytrue, preds))
    print("-"*50)
    print(confusion_matrix(ytrue, preds))
    print("-"*50)
    print(f"AUC = {roc_auc_score(ytrue, preds):.3f}")
    print("-"*50)

#%%
print_eval(ytest, cbc.predict(xtest) == 'True')

########################################################################
########################################################################
#%%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

si = SimpleImputer(strategy='most_frequent', add_indicator=True)
ss = StandardScaler()
# Add NA to encoder vocabulary for each column
oe = OrdinalEncoder(categories=[training[col].unique().tolist() + ['NA'] for col in catcols])

def start_pipe(data):
    return data.copy()

def standardize(data, train=True):
    numcols = data.columns.difference(catcols)
    data[numcols] = ss.fit_transform(data[numcols]) if train else ss.transform(data[numcols])
    return data

def label_encode(data, train=True):
    if train:
        data[catcols] = oe.fit_transform(data[catcols])
    else:
        for i, col in enumerate(catcols):
            # For unseen data: set unseen categories to NA
            data.loc[~data[col].isin(oe.categories_[i])] = 'NA'

        data[catcols] = oe.transform(data[catcols])
    return data
# %%


preped_xtrain = (xtrain
.pipe(start_pipe)
.pipe(standardize)
.pipe(label_encode)
)

preped_xval = (xval
.pipe(start_pipe)
.pipe(standardize, train=False)
.pipe(label_encode, train=False)
)

preped_xtest = (xtest
.pipe(start_pipe)
.pipe(standardize, train=False)
.pipe(label_encode, train=False)
)

#%%
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

tensor_xtrain = torch.Tensor(preped_xtrain.values)
tensor_ytrain = torch.Tensor(ytrain.values)

tensor_xval = torch.Tensor(preped_xval.values.astype('float32'))
tensor_yval = torch.Tensor(yval.values)

tensor_xtest = torch.Tensor(preped_xtest.values.astype('float32'))
tensor_ytest = torch.Tensor(ytest.values)



batch_size = 32
trainloader = DataLoader(TensorDataset(tensor_xtrain, torch.Tensor(ytrain.values)), shuffle=True, batch_size=batch_size)
# valloader = DataLoader(TensorDataset(tensor_xval, torch.Tensor(yval.values)))
# testloader = DataLoader(TensorDataset(tensor_xtest, torch.Tensor(ytest.values)))

#%%
net = nn.Sequential(
    nn.Linear(in_features=tensor_xtrain.size(-1), out_features=30),
    nn.ReLU(),
    nn.Linear(in_features=30, out_features=1),
    nn.Sigmoid()
)

optim = torch.optim.Adam(net.parameters())
criterion = nn.BCELoss()

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)

net.apply(init_weights)

#%%
from torch_lr_finder import LRFinder
lrf = LRFinder(net, optim, criterion)
lrf.range_test(trainloader, start_lr=10**-5, end_lr=1)
lrf.plot()
lrf.reset()

#%%
n_epochs = 20
scheduler = torch.optim.lr_scheduler.CyclicLR(optim, 10**-3, 10**-2, cycle_momentum=False)
history = {'train': [], 'val': []}

for epoch in range(n_epochs):
    for x, y in trainloader:
        yhat = net(x)
        loss = criterion(yhat, y)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
    
    with torch.no_grad():
        train_loss = criterion(net(tensor_xtrain), tensor_ytrain)
        val_loss = criterion(net(tensor_xval), tensor_yval)
        val_acc = (((net(tensor_xval)>0.5) == tensor_yval.view(-1, 1)).sum())/(tensor_xval.size(0)*1.0)

        history['train'].append(train_loss.item())
        history['val'].append(val_loss.item())

        print(f"Epoch #{epoch:3}: trainloss = {train_loss:.4f} & valloss = {val_loss:.4f} & val_acc = {val_acc:.3f}")

pd.DataFrame(history).plot()
print_eval(tensor_yval, net(tensor_xval).detach()>0.5)

#%%
catcols_idx = [xtrain.columns.get_loc(col) for col in catcols]
numcols_idx = [i for i in range(tensor_xtrain.size(-1)) if i not in catcols_idx]

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(
                num_embeddings=xtrain[col].cat.codes.max()+1,
                embedding_dim=max(xtrain[col].nunique()//3, 2),
            ) for col in catcols])
        
        self.hidden1 = nn.Linear(in_features=40, out_features=50)
        self.relu = nn.ReLU()
        self.out = nn.Linear(in_features=50, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        concatenated_embeddings = torch.cat([self.embeddings[i](x[:, colnr].long()) for i, colnr in enumerate(catcols_idx)], dim=1)
            
        #return concatenated_embeddings
        all_input = torch.cat([concatenated_embeddings, x[:, numcols_idx]], dim=1)
        
        x = self.relu(self.hidden1(all_input))
        out = self.sigmoid(self.out(x))
        
        return out
        



embnet = EmbeddingNet()


#%%
optim = torch.optim.Adam(embnet.parameters())
criterion = nn.BCELoss()
embnet.apply(init_weights)

#%%
from torch_lr_finder import LRFinder
lrf = LRFinder(embnet, optim, criterion)
lrf.range_test(trainloader, start_lr=10**-5, end_lr=1)
lrf.plot()
lrf.reset()

#%%
n_epochs = 15
scheduler = torch.optim.lr_scheduler.CyclicLR(optim, 10**-3, 10**-2, cycle_momentum=False)
history = {'train': [], 'val': []}

for epoch in range(n_epochs):
    for x, y in trainloader:
        yhat = embnet(x)
        loss = criterion(yhat, y)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
    
    with torch.no_grad():
        train_loss = criterion(embnet(tensor_xtrain), tensor_ytrain)
        val_loss = criterion(embnet(tensor_xval), tensor_yval)
        val_acc = (((embnet(tensor_xval)>0.5) == tensor_yval.view(-1, 1)).sum())/(tensor_xval.size(0)*1.0)

        history['train'].append(train_loss.item())
        history['val'].append(val_loss.item())

        print(f"Epoch #{epoch:3}: trainloss = {train_loss:.4f} & valloss = {val_loss:.4f} & val_acc = {val_acc:.3f}")

pd.DataFrame(history).plot()
print_eval(tensor_yval, embnet(tensor_xval).detach()>0.5)

#%%
loss_deltas = []
baseline = criterion(net(tensor_xtrain), tensor_ytrain).item()

for col in range(tensor_xtrain.size(1)):
    shuffled = tensor_xtrain.detach().clone()
    shuffled[:, col] = shuffled[torch.randperm(tensor_xtrain.size(0)), col]
    loss_deltas.append(criterion(net(shuffled), tensor_ytrain).item() - baseline)

pd.Series(loss_deltas, index=xtrain.columns).sort_values().plot(kind='barh')