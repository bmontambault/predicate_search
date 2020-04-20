
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as st
from time import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import itertools
from sklearn.manifold import TSNE

from context import predicate_search
from predicate_search import PredicateInduction, RobustNormal, Density, BottomUp, Predicate


# In[2]:


data = pd.read_csv('../data/satellite_s1.csv')


# In[3]:


data.columns = [f'f{col}' for col in data.columns]


# In[4]:


refit = True


# In[5]:


if refit:
    predicate_induction = PredicateInduction(data, [])
    with open('satellite_s1_pixal.pkl', 'wb') as f:
        pickle.dump(predicate_induction, f)
else:
    with open('satellite_s1_pixal.pkl', 'rb') as f:
        predicate_induction = pickle.load(f)


# In[6]:


all_features = ','.join(data.columns)


# In[7]:


distances = {all_features: predicate_induction.model.distance(predicate_induction.norm_data)}


# In[8]:


k=3


# In[9]:


for n in range(1, k+1):
    for group in itertools.combinations(data.columns, n):
        features = list(group)
        dist = predicate_induction.model.distance(predicate_induction.norm_data[features])
        distances[','.join(features)] = dist


# In[10]:


distances = pd.DataFrame(distances)


# In[11]:


distances


# In[12]:


threshold = 30


# In[13]:


targets_str = distances[distances > threshold].count().argmax()


# In[14]:


above_threshold = distances[targets_str][distances[targets_str] > threshold]


# In[15]:


targets = targets_str.split(',')


# In[16]:


targets


# In[17]:


tsne = TSNE(n_components=2)
projection = pd.DataFrame(tsne.fit_transform(data))


# In[18]:


def brush(x1, x2, y1, y2):
    fig, ax = plt.subplots()
    projection.plot.scatter(x=0, y=1, ax=ax)
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    index = list(projection[(projection[0] >= x1) & (projection[0] <= x2) &
                (projection[1] >= y1) & (projection[1] <= y2)].index)
    return index


# In[26]:


index = brush(-95, -60, -30, 7)


# In[27]:


len(index)


# In[28]:


index


# In[29]:


p = predicate_induction.find_predicates(targets=targets, c=.5, index=index, maxiters=2, topn=5)


# In[30]:


p


# In[31]:


data.plot.scatter(x='f33', y='f21')

