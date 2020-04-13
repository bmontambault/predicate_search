
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


# In[3]:


data = pd.read_csv('../data/breast_cancer_s1.csv')

