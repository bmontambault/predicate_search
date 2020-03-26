
# coding: utf-8

# In[1]:


from scipy.io import arff
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from context import predicate_search
from predicate_search import NormData, PredicateAnomalyDetection


# In[68]:


data = arff.loadarff('../data/Stamps/Stamps_withoutdupl_05_v10.arff')
df = pd.DataFrame(data[0])
df = df[df.att3<.3].reset_index(drop=True)
y = (df.outlier == b'yes').astype(int)
df = df.drop(['id', 'outlier'], axis=1)


# In[69]:


model = PredicateAnomalyDetection(c=.8, b=.1)
model.fit(df[['att3', 'att9']])


# In[70]:


predicates = model.search(targets=['att3', 'att9'])


# In[71]:


predicates


# In[72]:


sns.scatterplot(x='att3', y='att9', data=df.assign(label=y), hue='label')


# In[74]:


sns.pairplot(df.assign(label=y), hue='label', diag_kind=None)


# In[9]:


all_p = model.search()
disc_all_p = model.disc_predicates
for i in range(10):
    new_p = model.search(predicates=all_p)
    new_disc_p = model.disc_predicates
    all_p += new_p
    disc_all_p += new_disc_p


# In[19]:


att7 = [p for p in all_p if 'att6' in p.features]


# In[20]:


att7


# In[11]:


att7_merged = att7[0]
for p in att7[1:]:
    att7_merged = att7_merged.merge(p)


# In[14]:


all_p


# In[13]:


disc_all_p


# In[12]:


att7_merged

