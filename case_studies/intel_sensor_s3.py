
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as st
from time import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from context import predicate_search
from predicate_search import PredicateInduction, RobustNormal, Density, BottomUp, Predicate


# In[2]:


data = pd.read_csv('../data/intel_sensor_s3.csv')
data.dtime = pd.to_datetime(data.dtime)


# In[3]:


disc_cols = ['moteid']


# In[4]:


refit = False


# In[5]:


if refit:
    predicate_induction = PredicateInduction(data, disc_cols)
    with open('intel_sensor_s3_pixal.pkl', 'wb') as f:
        pickle.dump(predicate_induction, f)
else:
    with open('intel_sensor_s3_pixal.pkl', 'rb') as f:
        predicate_induction = pickle.load(f)


# In[6]:


distances = predicate_induction.model.distance(predicate_induction.norm_data[['temperature']])


# In[7]:


fig, ax = plt.subplots()
ax.plot(data.temperature, distances, 'bo');


# In[8]:


fig, ax = plt.subplots()
ax.plot(data.temperature, distances, 'bo')
rect = plt.Rectangle((115, 32), 15, 10, fill=False, facecolor='none', linestyle='--')
ax.add_patch(rect)


# In[10]:


index1 = list(data[data.temperature > 122].index)
p1 = predicate_induction.find_predicates(targets=['temperature'], c=.5, index=index1, maxiters=2, topn=5)


# In[11]:


for p in p1:
    print(p, '\n')


# In[12]:


fig, ax = plt.subplots()
ax.plot(data.temperature, distances, 'bo')
rect = plt.Rectangle((-50, 15), 20, 10, fill=False, facecolor='none', linestyle='--')
ax.add_patch(rect)


# In[13]:


index2 = list(data[data.temperature < -20].index)
p2 = predicate_induction.find_predicates(targets=['temperature'], c=.8, index=index2, maxiters=2, topn=5)


# In[14]:


for p in p2:
    print(p, '\n')

