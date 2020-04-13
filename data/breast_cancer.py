
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


# original breast cancer data: http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
# 
# download original brest cancer data: https://tufts.box.com/s/jub5u5et23u84x8qwrod0ptc1ueqwapx

# In[63]:


breast_cancer_original = pd.read_csv('breast_cancer_original.data', header=None)


# In[87]:


features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'nconcave',
            'symmetry', 'fractal']
measures = ['mean', 'ste', 'worst']
columns = ['index', 'label'] + ['_'.join(col) for col in list(itertools.product(features, measures))]


# In[90]:


breast_cancer_original.columns = columns


# In[91]:


breast_cancer_original.to_csv('breast_cancer.csv', index=False)


# processed data uploaded here: https://tufts.box.com/s/4dx7q3bsfag2xu3h5fsqah15r1fe56go

# ### Scenario 1

# Keep first 10 malignant cases as anomalies

# download breast cancer data: https://tufts.box.com/s/4dx7q3bsfag2xu3h5fsqah15r1fe56go

# In[94]:


breast_cancer = pd.read_csv('breast_cancer.csv')


# In[96]:


breast_cancer_s1 = pd.concat([breast_cancer[breast_cancer.label=='B'],
                              breast_cancer[breast_cancer.label=='M'].iloc[:10]])


# In[ ]:


brea

