
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


# In[100]:


raw_data = pd.read_csv('../data/breast_cancer.csv')
benign = raw_data[raw_data.diagnosis=='B']
malignant = raw_data[raw_data.diagnosis=='M']
df = pd.concat([benign, malignant.sample(10)])#.iloc[:10]])
y = (df.diagnosis == 'M').values.astype(int)
df = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).reset_index(drop=True)
df.columns = [c.replace(' ', '_') for c in df.columns]


# In[101]:


sns.pairplot(data=df.assign(label=y), hue='label', diag_kind=None)


# In[78]:


predicate_induction = PredicateInduction()


# In[79]:


predicate_induction.fit(df, how='t')


# In[50]:


targets = ['smoothness_worst', 'area_worst']


# In[80]:


all_distances = {f: predicate_induction.m.get_distances(predicate_induction.norm_data, features=[f])
                for f in df.columns}


# In[81]:


max(all_distances, key=lambda x: np.var(all_distances[x] > 10))


# In[82]:


all_distances['smoothness_worst'].max()


# In[83]:


distances = predicate_induction.m.get_distances(predicate_induction.norm_data, features=['concavity_mean'])


# In[84]:


d = pd.Series(distances).sort_values(ascending=False).reset_index(drop=True).reset_index()


# In[85]:


d.plot.scatter(x='index', y=0)


# In[54]:


threshold = 10


# In[58]:


tick = time()
predicates = predicate_induction.predicate_induction(targets=targets, threshold=threshold,
                                                     b=.001, c=.8, quantile=.75, verbose=True)
print(time() - tick)


# In[56]:


predicates


# In[31]:


predicates


# In[10]:


# max_vals = df.max()
# df = df / max_vals


# In[11]:


# model = PredicateAnomalyDetection(c=.1, b=1)
# model.fit(df)


# In[12]:


# p = model.search()


# In[13]:


# p[0]


# Worst Area, Worst Smoothness and Mean Texture

# In[93]:


sns.scatterplot(x='area_worst', y='radius_mean', data=df.assign(label=y), hue='label')


# In[15]:


# sns.pairplot(data=df[['smoothness_worst', 'texture_mean', 'area_worst']].assign(label=y), hue='label', diag_kind=None)


# In[95]:


sns.scatterplot(x='area_worst', y='perimeter_mean', data=df.assign(label=y), hue='label')


# In[90]:


sns.scatterplot(x='radius_worst', y='perimeter_worst', data=df.assign(label=y), hue='label')


# In[18]:


# a = pd.DataFrame(PCA(2).fit_transform(df[['area_worst', 'smoothness_worst', 'texture_mean']]))


# In[19]:


# sns.scatterplot(x=0, y=1, data=a.assign(label=y), hue='label')

