#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


# original breast cancer data: https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)

# download original satellite train data: https://tufts.box.com/s/dgueo97bahj6qnfxpmjwppqrx6648eow
#         
# download original satellite test data: https://tufts.box.com/s/6w25hwhewbsxt404kbmsturjf1k65kmo

# In[8]:


satellite_train = pd.read_csv('satellite_original.trn', header=None, delimiter=r"\s+")
satellite_test = pd.read_csv('satellite_original.trn', header=None, delimiter=r"\s+")
satellite_data = pd.concat([satellite_train, satellite_test])


# In[9]:


labels = dict([(int(a.split(' ')[0]), ' '.join(a.split(' ')[1:])) for a in '''1 red soil
2 cotton crop
3 grey soil
4 damp grey soil
5 soil with vegetation stubble
6 mixture class (all types present)
7 very damp grey soil'''.split('\n')])


# In[10]:


satellite_data[36] = satellite_data[36].map(labels)


# In[11]:


satellite_data.to_csv('satellite_data.csv', index=False)


# processed data uploaded here: https://tufts.box.com/s/mps252wo3zlkjmib5dc5wws9ralhu2hq

# ### Scenario 1
# 
# Red soil, gray soil, damp gray soil, and very damp soil taken as normal data. 75 random samples are added from cotton crop and vegetation stubble.
# 
# download setellite data: https://tufts.box.com/s/mps252wo3zlkjmib5dc5wws9ralhu2hq

# In[15]:


np.random.seed(574823)


# In[17]:


satellite_normal_s1 = satellite_data[satellite_data[36].isin(
    ['red soil', 'gray soil', 'damp gray soil', 'very damp gray soil'])].assign(label='0')
satellite_anomalies_s1 = satellite_data[satellite_data[36].isin(['cotton crop', 'vegetation stubble']
                                                         )].assign(label='1').sample(75)


# In[22]:


satellite_s1 = pd.concat([satellite_normal_s1, satellite_anomalies_s1]).drop([36, 'label'], axis=1)


# In[24]:


satellite_s1.head()


# In[25]:


satellite_s1.to_csv('satellite_s1.csv', index=False)


# In[ ]:




