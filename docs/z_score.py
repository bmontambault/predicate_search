
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis

from context import predicate_search
from predicate_search import RobustNormal


# In[2]:


raw_data = pd.read_csv('../data/breast_cancer.csv')
benign = raw_data[raw_data.diagnosis=='B']
malignant = raw_data[raw_data.diagnosis=='M']
df = pd.concat([benign, malignant.iloc[:10]])
y = (df.diagnosis == 'M').values.astype(int)
df = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).reset_index(drop=True)
df.columns = [c.replace(' ', '_') for c in df.columns]


# In[3]:


max_vals = df.max()
df = df / max_vals


# In[56]:


df.head()


# In[4]:


model = RobustNormal()
model.fit(df)


# In[11]:


mean = pd.Series(model.mean, index=df.columns)
cov = pd.DataFrame(model.cov, index=df.columns, columns=df.columns)


# In[50]:


def get_z_scores(df, mean, cov, k=2):
    res = {}
    for n in range(2, k+1):
        for group in itertools.combinations(df.columns, n):
            features = list(group)
            m = mean[features].values
            c = cov[features].loc[features].values
            prec = np.linalg.inv(c)
            x = df[features].values
            
            VI = np.linalg.inv(c)
            dist = np.diag(np.sqrt(np.dot(np.dot((x-m),VI),(x-m).T)))
            res[group] = dist.tolist()
    return res


# In[51]:


a = get_z_scores(df, mean, cov)


# In[53]:


a[('radius_mean', 'texture_mean')]


# In[19]:


for i in range(1, 4):
    for j in itertools.combinations(df.columns, i):
        print(j)


# In[15]:


for i in itertools.combinations(df.columns, 1):
    print(i)

