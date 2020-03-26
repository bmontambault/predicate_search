
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import scipy.stats as st
from sklearn.covariance import MinCovDet
from time import time

from context import predicate_search
from predicate_search import PredicateInduction, PredicateData, PredicateSearch, ContBasePredicate, DiscBasePredicate, CompoundPredicate, RobustNormal


# In[2]:


raw_data = pd.read_csv('../data/sensor_data.csv')
raw_data.dtime = pd.to_datetime(raw_data.dtime)
startdate = '2004-2-28'
enddate = '2004-3-3'
df = raw_data[(raw_data.dtime>=startdate) & (raw_data.dtime<=enddate)].reset_index(drop=True)


# In[3]:


predicate_induction = PredicateInduction()


# In[15]:


predicate_induction.fit(df, how='t', disc_cols=['moteid'])


# In[16]:


targets = ['temperature']


# In[17]:


distances = predicate_induction.m.get_distances(predicate_induction.norm_data, targets)


# In[18]:


d = pd.Series(distances).sort_values(ascending=False).reset_index(drop=True).reset_index()


# In[19]:


d.plot.scatter(x='index', y=0)


# In[20]:


threshold = 30


# In[31]:


tick = time()
predicates = predicate_induction.predicate_induction(targets, threshold, c=1, quantile=.25, verbose=True)
print(time() - tick)


# In[32]:


predicates


# In[33]:


tick = time()
predicates = predicate_induction.predicate_induction(targets, threshold=None, c=1, quantile=.25, verbose=True)
print(time() - tick)


# In[28]:


predicates

