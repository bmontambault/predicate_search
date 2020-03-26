
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from sklearn.decomposition import PCA

from context import predicate_search
from predicate_search import NormData, PredicateAnomalyDetection, SearchData


# In[21]:


train_data = pd.read_csv('../data/pendigits.tra', header=None)
test_data = pd.read_csv('../data/pendigits.tes', header=None)


# In[22]:


raw_data = pd.concat([train_data, test_data])


# In[45]:


inliers = train_data[train_data[16] == 8]
outliers = train_data[train_data[16] != 8].iloc[:10]
df = pd.concat([inliers, outliers]).reset_index(drop=True)
y = (df[16] != 8).astype(int).values
df = df.drop(16, axis=1)


# In[46]:


sns.pairplot(data=df.assign(label=y), hue='label', diag_kind=None)

