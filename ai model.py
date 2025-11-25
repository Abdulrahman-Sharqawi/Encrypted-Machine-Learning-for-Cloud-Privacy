#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from pandas import DataFrame 
import matplotlib.pyplot as plt
import time

start_time = time.time()
rawData = pd.read_csv("water_requirement_sa.csv") 

rawData.describe() 


# In[2]:


rawData.head(100) 


# In[3]:


def catigorize (raw) : 
    if (raw['WATER REQUIREMENT'] > 50) : 
        return 1 
    elif (raw['WATER REQUIREMENT'] > 40) : 
        return 2 
    elif (raw['WATER REQUIREMENT'] > 30) : 
        return 3 
    elif (raw['WATER REQUIREMENT'] > 20) : 
        return 4 
    elif (raw['WATER REQUIREMENT'] > 10) : 
        return 5 
    else : 
        return 6

#rawData ['irregation catigory'] = rawData.apply(catigorize, axis=1)

#rawData.to_csv('water_requirement_sa.csv', index= False) 

#uniqueValuesOfCrops = rawData['CROP TYPE'].unique() 



#uniqueValuesOfSoil = rawData['SOIL TYPE'].unique()
 

uniqueValuesOfRegion = rawData['REGION'].unique() 

#rawData['CROP TYPE'] = rawData['CROP TYPE'].map({'BANANA' : 1, 'SOYABEAN': 2, 'CABBAGE': 3, 'POTATO': 4, 'RICE': 5, 'MELON' : 6, 'MAIZE': 7, 'CITRUS' : 8, 'BEAN' : 9, 'WHEAT' : 10, 'MUSTARD': 11, 'COTTON' : 12, 'SUGARCANE' : 13, 'TOMATO' : 14, 'ONION' : 15 }) 

#uniqueValuesOfCrops
#uniqueValuesOfSoil
uniqueValuesOfRegion 


# In[4]:


rawData['SOIL TYPE'] = rawData['SOIL TYPE'].map({'DRY' : 4, 'HUMID' : 2, 'WET' : 0}) 


# In[5]:


rawData['REGION'] = rawData['REGION'].map({'DESERT': 3, 'SEMI ARID': 2, 'SEMI HUMID' : 1, 'HUMID' : 0}) 

rawData.head(30) 


# In[6]:


rawData['TEMPERATURE'] = rawData['TEMPERATURE'].map({'10-20' : 15, '20-30' : 25, '30-40': 35, '40-50' : 45}) 
rawData.head(400) 


# In[7]:


rawData['WEATHER CONDITION'] = rawData['WEATHER CONDITION'].map({'RAINY' : 0, 'WINDY': 1, 'NORMAL': 3, 'SUNNY': 4}) 
rawData.head(20) 


# In[8]:


numericColmuns = []
numericColmuns.extend(list(rawData.dtypes[rawData.dtypes == np.int64].index))
numericColmuns.extend(list(rawData.dtypes[rawData.dtypes == np.float64].index)) 
numericColmuns


# In[9]:


numericColmuns.remove('Farmer_Age')
numericColmuns.remove('Annual_Income')
numericColmuns.remove('Water_Bill')

numericColmuns


# In[10]:


numericData = DataFrame(rawData, columns=numericColmuns) 


# In[11]:


numericData.describe() 


# In[12]:


numericData.head(10) 


# In[ ]:


# get_ipython().system('pip install torch torchvision torchaudio')
# replaced with a runtime check (so script works when run outside Jupyter)
try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is not installed in this Python environment.\n"
        "Install it in a terminal, for example (CPU-only):\n"
        "  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
        "Or see https://pytorch.org/get-started/locally/ for CUDA-enabled wheels."
    )


# In[ ]:


import torch 
import torch.nn as nn


# In[ ]:


numeric_x_columns = list(numericData.columns)
numeric_x_columns.remove('WATER REQUIREMENT')
numeric_y_columns = ['WATER REQUIREMENT'] 


# In[ ]:


numeric_x_columns


# In[ ]:


numeric_x_df = DataFrame(numericData, columns=numeric_x_columns) 
numeric_y_df = DataFrame(numericData, columns=numeric_y_columns)


# In[ ]:


numeric_x = torch.tensor(numeric_x_df.values, dtype=torch.float)
numeric_y = torch.tensor(numeric_y_df.values, dtype=torch.float)


# In[ ]:


numeric_x.shape


# In[ ]:


numeric_y.shape 


# In[ ]:


class Net (nn.Module) : 
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(Net, self).__init__()
        
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)
        
    def forward(self, x):
        y_pred = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(y_pred).clamp(min=0)
        y_pred = self.linear3(y_pred).clamp(min=0)
        y_pred = self.linear4(y_pred)
        return y_pred


# In[ ]:


H1, H2, H3 = 500, 1000, 200 


# In[ ]:


D_in, D_out = numeric_x.shape[1], numeric_y.shape[1] 


# In[ ]:


model1 = Net(D_in, H1, H2, H3, D_out)


# In[ ]:


criterion = nn.MSELoss(reduction='sum') 


# In[ ]:


optimizer = torch.optim.SGD(model1.parameters(), lr=1e-4) 


# In[ ]:


losses1 = []

for t in range(500):
    y_pred = model1(numeric_x)
    
    loss = criterion(y_pred, numeric_y)
    print(t, loss.item())
    losses1.append(loss.item())
    
    if torch.isnan(loss):
        break
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[ ]:


means, maxs, mins = dict(), dict(), dict() 


# In[ ]:


for col in numericData:
    means[col] = numericData[col].mean()
    maxs[col] = numericData[col].max()
    mins[col] = numericData[col].min() 


# In[ ]:


numericData = (numericData - numericData.mean()) / (numericData.max() - numericData.min())  


# In[ ]:


numeric_x_df = DataFrame(numericData, columns=numeric_x_columns)
numeric_y_df = DataFrame(numericData, columns=numeric_y_columns) 


# In[ ]:


numeric_x = torch.tensor(numeric_x_df.values, dtype=torch.float)
numeric_y = torch.tensor(numeric_y_df.values, dtype=torch.float)


# In[ ]:


model2 = Net(D_in, H1, H2, H3, D_out) 


# In[ ]:


criterion = nn.MSELoss(reduction='sum') 


# In[ ]:


optimizer = torch.optim.SGD(model2.parameters(), lr=1e-4)


# In[ ]:


losses2 = []

for t in range(600):
    y_pred = model2(numeric_x)
    
    loss = criterion(y_pred, numeric_y)
    print(t, loss.item())
    losses2.append(loss.item())
    
    if torch.isnan(loss):
        break
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[ ]:


model3 = Net(D_in, H1, H2, H3, D_out)
optimizer = torch.optim.SGD(model3.parameters(), lr=1e-4 * 2)


# In[ ]:


losses3 = []

for t in range(500):
    y_pred = model3(numeric_x)
    
    loss = criterion(y_pred, numeric_y)
    print(t, loss.item())
    losses3.append(loss.item())
    
    if torch.isnan(loss):
        break
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[ ]:


model4 = Net(D_in, H1, H2, H3, D_out)
optimizer = torch.optim.Adam(model4.parameters(), lr=1e-4 * 2)


# In[ ]:


losses4 = []

for t in range(500):
    y_pred = model4(numeric_x)
    
    loss = criterion(y_pred, numeric_y)
    print(t, loss.item())
    losses4.append(loss.item())
    
    if torch.isnan(loss):
        break
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[ ]:


result = DataFrame(y_pred.data.numpy(), columns=['Predicted water requirnment'])  


# In[ ]:


result 


# In[ ]:


result['Predicted water requirnment'] = result['Predicted water requirnment'].fillna(0) 


# In[ ]:


result


# In[ ]:


result['Predicted water requirnment'] = result['Predicted water requirnment'] * (maxs['WATER REQUIREMENT'] - mins['WATER REQUIREMENT']) + means['WATER REQUIREMENT']


# In[ ]:


result


# In[ ]:


result.to_csv('./water_requirement_sa_result.csv', columns=['Predicted water requirnment'], index= False)


# In[ ]:




end_time = time.time()
total_time = end_time - start_time

print("\nTraining completed in:", total_time, "seconds")
print("Training completed in:", total_time/60, "minutes")
print("Training completed in:", total_time/3600, "hours")
