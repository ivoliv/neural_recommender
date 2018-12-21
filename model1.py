
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


path = '/Users/ivoliv/data/yoochoose'
NUMROWS = 1e6


# In[3]:


get_ipython().system('ls -l $path')


# In[4]:


def print_uniques(df):
    name =[x for x in globals() if globals()[x] is df][0]
    print('== {} =='.format(name))
    print('{:<15}: {:,}'.format('total obs', len(df)))
    print('unique values:')
    for c in df.columns:
        print(' {:<14}: {:,}'.format(c, len(df[c].unique())))

def filter_sessions(table, filter_table):
    filter_sessions_ = filter_table['sessionID'].unique()
    table = table[table['sessionID'].isin(filter_sessions_)]
    return table.sort_values(by=['sessionID', 'timestamp'])


# In[5]:


clicks = pd.read_csv(os.path.join(path, 'yoochoose-clicks.dat'), header=None, nrows=NUMROWS,
                     names=['sessionID', 'timestamp', 'itemID', 'category'])
buys = pd.read_csv(os.path.join(path, 'yoochoose-buys.dat'), header=None, nrows=NUMROWS,
                   names=['sessionID', 'timestamp', 'itemID', 'price', 'quantity'])


# ### Only keep buy sessions with positive number of items

# In[6]:


buys = buys[buys['quantity'] > 0]
buys=buys.sort_values(by=['sessionID', 'timestamp'])


# ### Only keep click sessions related to positive buys

# In[7]:


clicks = filter_sessions(clicks, buys)
print_uniques(clicks)


# ### Filter buys to only relate to remaining (led to positive) clicks

# In[8]:


buys = filter_sessions(buys, clicks)
print_uniques(buys)


# In[9]:


# Click count statistics
clicks.groupby('sessionID')['sessionID'].count().describe()


# In[10]:


clicks[clicks['sessionID'] == 11]


# In[11]:


buys[buys['sessionID'] == 11]


# In[12]:


import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import tqdm as tqdm


# In[13]:


class SessionDataset(Dataset):
    def __init__(self, clicks_df, buys_df, 
                 train_split=0.7, dev_test_split=0.5, random_state=123):
        sessionIDs = clicks['sessionID'].unique()
        
        assert sum(sessionIDs != buys['sessionID'].unique()) == 0,             "sessionIDs of clicks and buys have to match, be sure to filter and sort by sessions ID"
    
        print('Indexing... ', flush=True, end='')
        sessionIDs = clicks['sessionID'].unique()
        clicks_df['idx'] = clicks['sessionID'].apply(lambda x: np.argwhere(sessionIDs==x)[0][0])
        buys_df['idx'] = buys['sessionID'].apply(lambda x: np.argwhere(sessionIDs==x)[0][0])
        indices = clicks['idx'].unique()
        print('Done.', flush=True)
        
        print('Processing clicks... ', flush=True, end='')
        clicks_items = {}
        for i in clicks['idx'].unique():
            clicks_items[i] = list(clicks_df[clicks_df['idx'] == i]['itemID'])
        print('Done.', flush=True)

        print('Processing buys... ', flush=True, end='')
        buys_items = {}
        for i in buys['idx'].unique():
            buys_items[i] = list(buys_df[buys_df['idx'] == i]['itemID'])
        print('Done.', flush=True)
        
        self.items = np.unique(np.concatenate([clicks['itemID'], buys['itemID']]))
        self.item_to_i= {}
        self.i_to_item= {}
        for i, k in enumerate(self.items):
            self.item_to_i[k] = i
            self.i_to_item[i] = k
        self.nitems = len(self.items)
        
        self.X_train, X_left = train_test_split(
            indices, test_size=1-train_split, random_state=random_state, shuffle=True)

        self.X_val, self.X_test = train_test_split(
            X_left, test_size=dev_test_split, random_state=random_state, shuffle=True)
        
        self.clicks_items_train = {i: clicks_items[k] for i,k in enumerate(self.X_train)}
        self.clicks_items_val = {i: clicks_items[k] for i,k in enumerate(self.X_val)}
        self.clicks_items_test = {i: clicks_items[k] for i,k in enumerate(self.X_test)}
 
        self.buys_items_train = {i: buys_items[k] for i,k in enumerate(self.X_train)}
        self.buys_items_val = {i: buys_items[k] for i,k in enumerate(self.X_val)}
        self.buys_items_test = {i: buys_items[k] for i,k in enumerate(self.X_test)}
        
        self._lookup_dict = {'train': (self.clicks_items_train, self.buys_items_train),
                             'val': (self.clicks_items_val, self.buys_items_val),
                             'test': (self.clicks_items_test, self.buys_items_test)}
        
        self.set_split('train')
        
    def __len__(self):
        return len(self._lookup_dict[self._split][0])
        
    def set_split(self, split):
        self._split = split
    
    def __getitem__(self, index):
        clicks_items, buys_items = self._lookup_dict[self._split]
        
        one_hot_clicks = np.zeros(self.nitems)
        
        for item in clicks_items[index]:
            one_hot_clicks[self.item_to_i[item]] = 1
            
        first_item_bought = self.item_to_i[buys_items[index][0]]
        
        return {'idx': index,
                'x_data': torch.tensor(one_hot_clicks).float(),
                'y_target': torch.tensor(first_item_bought).view(-1)}
    
    def getitem_raw(self, index):
        clicks_items, buys_items = self._lookup_dict[self._split]
        
        return {'idx': index,
                'x_data': clicks_items[index],
                'y_target': buys_items[index]}
        
    def __str__(self):
        strval = 'sessions in ' + self._split + ': {:,}'.format(len(self._lookup_dict[self._split][0])) + '\n'
        strval += 'items in ' + self._split + ': {:,}'.format(self.nitems)
        return strval


# In[14]:


dataset = SessionDataset(clicks, buys)


# In[15]:


print(dataset)


# In[16]:


batch_size = 16
dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                        drop_last=False, shuffle=True)


# In[17]:


for i in dataloader:
    idx = i['idx'][0].item()
    print(i['idx'][0])
    print(sum(i['x_data'][0]))
    print(i['y_target'][0])
    print(i)
    break


# In[18]:


idx


# In[19]:


clicks[clicks['idx'] == dataset.X_train[idx]]


# In[20]:


buys[buys['idx'] == dataset.X_train[idx]]


# In[21]:


from torch import nn
import torch.nn.functional as F


# In[22]:


class productMLP(nn.Module):
    def __init__(self, input_size, h_sizes, output_size):
        super(productMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, h_sizes[0])
        
        self.hidden = nn.ModuleList()
        for i in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[i], h_sizes[i+1]))
            
        self.last = nn.Linear(h_sizes[i], output_size)
        
    def forward(self, inputs):
        
        x = F.relu(self.fc1(inputs))
        for h in self.hidden:
            x = F.relu(h(x))
        output = self.last(x)
        
        return output


# In[23]:


model = productMLP(dataset.nitems, [100,100], dataset.nitems)


# In[24]:


print(model)


# In[25]:


data = next(iter(dataloader))


# In[26]:


x_data = data['x_data']
y_target = data['y_target']


# In[27]:


y_predict = model(x_data)
y_predict


# In[28]:


ce_loss = nn.CrossEntropyLoss()


# In[29]:


y_target.view(-1)


# In[30]:


y_predict.shape


# In[31]:


loss = ce_loss(y_predict, y_target.view(-1))


# In[32]:


loss


# In[33]:


from torch.optim import Adam


# In[34]:


optim = Adam(params=model.parameters(), lr=0.01)
batch_size = 16


# In[35]:


def run_batch(model, runtype='train', device=torch.device('cpu')):
    
    model.to(device)
    
    if runtype == 'train':
        model.train()
    dataset.set_split(runtype)
    batch_size = 16
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            drop_last=False, shuffle=True)
    
    running_loss = 0
    
    for data in dataloader:

        model.zero_grad()
        
        x_data = data['x_data'].to(device)
        y_target = data['y_target'].to(device)
    
        y_predict = model(x_data)
        
        loss = ce_loss(y_predict, y_target.view(-1))
        
        running_loss += loss.item()
        
        loss.backward()
        
        optim.step()
        
    running_loss /= len(dataset)
    
    return running_loss


# In[ ]:


EPOCHS = 30
hist = []
for epoch in range(EPOCHS):
    loss_train = run_batch(model, 'train')
    loss_val = run_batch(model, 'val')
    print('{} {:.4} {:.4}'.format(epoch+1, loss_train, loss_val))
    hist.append((loss_train, loss_val))


# In[ ]:


hist = np.array(hist)
plt.plot(hist[:,0], 'r', hist[:,1], 'g')

