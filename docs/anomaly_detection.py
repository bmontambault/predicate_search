
# coding: utf-8

# In[1]:


import pymc3 as pm
import scipy.stats as st
from scipy.special import binom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

from context import predicate_search
from predicate_search import NormData, PredicateAnomalyDetection


# In[2]:


def precision(y, predy):
    true_positives = true_positives = ((y == 1).astype(int) * (predy == 1).astype(int)).sum()
    if predy.sum() == 0:
        return 0
    else:
        return true_positives / predy.sum()

def recall(y, predy):
    true_positives = ((y == 1).astype(int) * (predy == 1).astype(int)).sum()
    if y.sum() == 0:
        return 0
    else:
        return true_positives / y.sum()

def f1(y, predy):
    p = precision(y, predy)
    r = recall(y, predy)
    if p + r == 0:
        return 0
    else:
        return 2 * (p*r) / (p + r)

def accuracy(y, predy):
    return ((predy == y).astype(int).mean()) 


# In[3]:


class TestModel(object):
    
    def __init__(self, model, params, eval_funcs=[precision, recall, f1, accuracy]):
        self.model = model
        self.params = params
        self.eval_funcs = eval_funcs
        self.all_params = [dict(zip(self.params.keys(), v)) for v in itertools.product(*self.params.values())]
        
    def score_param(self, X, y, param_dict):
        m = self.model(**param_dict)
        predy = m.fit_predict(X)
        for f in self.eval_funcs:
            param_dict[f.__name__] = f(y, predy)
        param_dict['model'] = m
        return param_dict
    
    def score(self, X, y):
        return pd.DataFrame([self.score_param(X, y, param_dict) for param_dict in self.all_params])


# In[4]:


class TestNormal:
    
    def __init__(self, params):
        self.params = params
        
    def test(self, param_dict, models, model_params, n_samples):
        all_res = []
        for i in range(n_samples):
            norm_data = NormData(**param_dict)
            for j in range(len(models)):
                model = models[j]
                params = model_params[j]
                test = TestModel(model, params)
                res = test.score(norm_data.tainted, norm_data.y)
                res['data'] = norm_data
                all_res.append(res)
        return all_res


# In[22]:


# norm_params = {'n': [1000], 'm': [2]}
# model_params = {'c': np.linspace(.1, 1, 10), 'b': [0, .1, .2]}
# test_normal = TestNormal(norm_params)
# res = pd.concat(test_normal.test({'n': 1000, 'm': 2}, [PredicateAnomalyDetection], [model_params], 10))


# In[21]:


# res = pd.concat(r)


# In[7]:


# norm_data = NormData(1000, 2, k=2)
# data = norm_data.tainted
# y = norm_data.y
# norm_data.predicate


# In[8]:


# norm_data.plot2d()


# In[9]:


# m = PredicateAnomalyDetection()
# m.fit(data)
# m.search()


# In[10]:


# params = {'c': np.linspace(.1, 1, 10), 'b': [0, .1, .2]}
# test = TestModel(PredicateAnomalyDetection, params)
# res = test.score(data, y)


# In[11]:


# pd.concat(res)

