
# coding: utf-8

# In[130]:


import pandas as pd
import numpy as np
import scipy.stats as st
from time import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import itertools
from sklearn.manifold import TSNE

from context import predicate_search
from predicate_search import PredicateInduction, RobustNormal, Density, BottomUp, Predicate


# In[21]:


data = pd.read_csv('../data/breast_cancer_s1.csv')


# In[22]:


refit = False


# In[23]:


if refit:
    predicate_induction = PredicateInduction(data, [])
    with open('breast_cancer_s1_pixal.pkl', 'wb') as f:
        pickle.dump(predicate_induction, f)
else:
    with open('breast_cancer_s1_pixal.pkl', 'rb') as f:
        predicate_induction = pickle.load(f)


# In[24]:


all_features = ','.join(data.columns)


# In[25]:


distances = {all_features: predicate_induction.model.distance(predicate_induction.norm_data)}


# In[26]:


k=3


# In[27]:


for n in range(1, k+1):
    for group in itertools.combinations(data.columns, n):
        features = list(group)
        dist = predicate_induction.model.distance(predicate_induction.norm_data[features])
        distances[','.join(features)] = dist


# In[28]:


distances = pd.DataFrame(distances)


# In[45]:


distances


# In[118]:


threshold = 40


# In[119]:


targets_str = distances[distances > threshold].count().argmax()


# In[120]:


above_threshold = distances[targets_str][distances[targets_str] > threshold]


# In[121]:


targets = targets_str.split(',')


# In[128]:


tsne = TSNE(n_components=2)
projection = pd.DataFrame(tsne.fit_transform(data))


# In[129]:


projection.plot.scatter(x=0, y=1)


# In[132]:


data.iloc[-10:]


# In[138]:


# sns.pairplot(data=data.assign(label=data.index>=357), hue='label', diag_kind=None)


# In[124]:


index = list(above_threshold.index)


# In[125]:


index


# In[126]:


len(index)


# In[136]:


index = data[data.index>=157].index


# In[137]:


p = predicate_induction.find_predicates(targets=targets, c=.4, quantile=.25, index=index, maxiters=1,
                                         topn=5)


# In[64]:


for pi in p:
    print(pi, '\n')


# In[65]:


data[data.perimeter_mean == .4108]

