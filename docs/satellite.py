
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


labels = dict([(int(a.split(' ')[0]), ' '.join(a.split(' ')[1:])) for a in '''1 red soil
2 cotton crop
3 grey soil
4 damp grey soil
5 soil with vegetation stubble
6 mixture class (all types present)
7 very damp grey soil'''.split('\n')])


# In[3]:


train = pd.read_csv('../data/sat.trn', header=None, delimiter=r"\s+")
test = pd.read_csv('../data/sat.trn', header=None, delimiter=r"\s+")
train[36] = train[36].map(labels)
test[36] = test[36].map(labels)
raw_data = pd.concat([train, test])


# In[4]:


normal = raw_data[raw_data[36].isin(['red soil', 'gray soil', 'damp gray soil', 'very damp gray soil'])
                 ].assign(label='0')
anomalies = raw_data[raw_data[36].isin(['cotton crop', 'vegetation stubble'])].assign(label='1').sample(75)


# In[5]:


df = pd.concat([normal, anomalies]).drop(36, axis=1)


# In[6]:


sns.pairplot(data=df, hue='label', diag_kind=None)

