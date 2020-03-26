
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree

from context import predicate_search
from predicate_search import PredicateData, PredicateSearch, TDPredicateSearch, ContBasePredicate


# In[2]:


class SynthSingleTarget:
    
    def __init__(self, n=2000, m=2, att_min=0, att_max=100, mu=80, med_outlier_lower=20, med_outlier_upper=80,
                 high_outlier_lower=40, high_outlier_upper=60):
        self.n = n
        self.m = m
        self.att_min = att_min
        self.att_max = att_max
        self.mu = mu
        self.med_outlier_lower = med_outlier_lower
        self.med_outlier_upper = med_outlier_upper
        self.high_outlier_lower = high_outlier_lower
        self.high_outlier_upper = high_outlier_upper
        
    def generate_attributes(self, n, m, att_min, att_max):
        attributes = pd.DataFrame(np.random.uniform(att_min, att_max, size=(n, m)))
        attributes.columns = [f"f{col}" for col in attributes.columns]
        return attributes
        
    def generate_target(self, attributes, mu):
        att = list(attributes.columns)
        attributes['med_outlier'] = np.all(attributes[att] >= self.med_outlier_lower, axis=1
                                          ) & np.all(attributes[att] <= self.med_outlier_upper, axis=1)
        attributes['high_outlier'] = np.all(attributes[att] >= self.high_outlier_lower, axis=1
                                           ) & np.all(attributes[att] <= self.high_outlier_upper, axis=1)
        attributes['outlier_type'] = attributes.med_outlier.astype(int) + attributes.high_outlier.astype(int)
        attributes['mu'] = attributes.outlier_type.map({0: 10, 1: (mu + 10)/2, 2: mu})
        attributes['val'] = np.random.normal(attributes.mu, 10)
        
        return attributes[att + ['val']], attributes.med_outlier, attributes.high_outlier
    
    def generate_data(self):
        attributes = self.generate_attributes(self.n, self.m, self.att_min, self.att_max)
        data, med_outlier, high_outlier = self.generate_target(attributes, self.mu)
        return data, med_outlier, high_outlier


# In[3]:


class NormAnomalyDetection:
    
    def __init__(self, mean, var, c=.5):
        self.mean = mean
        self.var = var
        self.c = c
        
    def predict(self, data, c=None):
        if c is None:
            c = self.c
        lower, upper = st.norm(self.mean, self.var).interval(c)
        predy = ((data.val < lower) | (data.val > upper)).astype(int)
        return predy


# In[4]:


class PredAnomalyDetection:
    
    def __init__(self, mean, var, c=.5, b=.1, quantile=.25):
        self.mean = mean
        self.var = var
        self.c = c
        self.b = b
        self.quantile = quantile
        
    def search(self, data, c=None, b=None, quantile=None):
        if c is None:
            c = self.c
        if b is None:
            b = self.b
        if quantile is None:
            quantile = self.quantile
        predicate_data = PredicateData(data)
        disc_data = predicate_data.disc_data
        logp = st.norm(self.mean, self.var).logpdf(data.val)
        predicates = predicate_data.get_base_predicates(logp)
        predicate_search = PredicateSearch(predicates)
        predicate = predicate_search.search_features(features=data.drop('val',axis=1).columns, c=c, b=b,
                                                     quantile=quantile)
        return predicate_search, predicate
        
    def predict(self, data, c=None, b=None, quantile=None):
        predicate_search, predicate = self.search(data, c, b, quantile)
        predy = data.index.isin(predicate.selected_index).astype(int)
        return predicate_search, predicate, predy


# In[59]:


class Evaluate:
    
    def evaluate_predy(self, y, predy):
        true_positives = true_positives = ((y == 1).astype(int) * (predy == 1).astype(int)).sum()
        if predy.sum() > 0:
            precision = true_positives / predy.sum()
        else:
            precision = 0
        if y.sum() > 0:
            recall = true_positives / y.sum()
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * (precision*recall) / (precision + recall)
        else:
            f1 = 0
        return precision, recall, f1
    
    def evaluate_data(self, mean, var, synth, data, med_outlier, high_outlier, n=50):
        results = []
        pred = PredAnomalyDetection(mean, var)
        norm = NormAnomalyDetection(mean, var)
        for c in np.linspace(0, 1, n):
            print(c)
            predicate_search, predicate, pred_y = pred.predict(data, c=c)
            norm_y = norm.predict(data, c=c)
            
            pred_med_precision, pred_med_recall, pred_med_f1 = self.evaluate_predy(med_outlier, pred_y)
            norm_med_precision, norm_med_recall, norm_med_f1 = self.evaluate_predy(med_outlier, norm_y)
            pred_high_precision, pred_high_recall, pred_high_f1 = self.evaluate_predy(high_outlier, pred_y)
            norm_high_precision, norm_high_recall, norm_high_f1 = self.evaluate_predy(high_outlier, norm_y)
            
            results.append([c, pred_med_precision, pred_med_recall, pred_med_f1,
                               norm_med_precision, norm_med_recall, norm_med_f1,
                               pred_high_precision, pred_high_recall, pred_high_f1,
                               norm_high_precision, norm_high_recall, norm_high_f1])
            
        results = pd.DataFrame(results)
        results.columns = ['c', 'pred_med_precision', 'pred_med_recall', 'pred_med_f1',
                               'norm_med_precision', 'norm_med_recall', 'norm_med_f1',
                               'pred_high_precision', 'pred_high_recall', 'pred_high_f1',
                               'norm_high_precision', 'norm_high_recall', 'norm_high_f1']
        return results
    
    def evaluate_norm(self, mu, m, mean, var, n_samples):
        all_results = []
        synth = SynthSingleTarget(mu=mu, m=m)
        for i in range(n_samples):
            print(i)
            self.data, med_outlier, high_outlier = synth.generate_data()
            all_results.append(evaluate.evaluate_data(mean, var, synth, self.data, med_outlier, high_outlier))
        return pd.concat(all_results)


# In[60]:


synth = SynthSingleTarget()
data = synth.generate_data()[0]


# In[58]:


sns.scatterplot(x='f0', y='f1', hue='val', data=data)


# In[62]:


evaluate = Evaluate()


# In[46]:


# easy_2d_results = evaluate.evaluate_norm(80, 2, 10, 10, 100)
easy_2d_results = pd.read_csv('easy_2d_results_v2.csv')


# In[64]:


sns.lineplot(x="c", y="pred_high_precision", data=easy_2d_results)
sns.lineplot(x="c", y="pred_med_precision", data=easy_2d_results)


# In[65]:


sns.lineplot(x="c", y="pred_high_recall", data=easy_2d_results)
sns.lineplot(x="c", y="pred_med_recall", data=easy_2d_results)


# In[47]:


sns.lineplot(x="c", y="pred_high_f1", data=easy_2d_results)
sns.lineplot(x="c", y="pred_med_f1", data=easy_2d_results)


# In[48]:


# hard_2d_results = evaluate.evaluate_norm(40, 2, 10, 10, 100)
hard_2d_results = pd.read_csv('hard_2d_results_v2.csv')


# In[49]:


sns.lineplot(x="c", y="pred_high_f1", data=hard_2d_results)
sns.lineplot(x="c", y="pred_med_f1", data=hard_2d_results)


# In[63]:


easy_3d_results = evaluate.evaluate_norm(80, 3, 10, 10, 100)


# In[8]:


data = evaluate.data


# In[9]:


predicate_data = PredicateData(data)
disc_data = predicate_data.disc_data
logp = st.norm(10, 10).logpdf(data.val)
predicates = predicate_data.get_base_predicates(logp)
predicate_search = PredicateSearch(predicates)


# In[10]:


predicate = predicate_search.search_features(features=data.drop('val',axis=1).columns, c=0.6444444444444444,
                                             verbose=True)


# In[11]:


# synth = SynthSingleTarget(mu=80, m=2)
# data, med_outlier, high_outlier = synth.generate_data()
# mean = 10
# var = 10


# In[12]:


# predicate_data = PredicateData(data)
# disc_data = predicate_data.disc_data
# logp = st.norm(mean, var).logpdf(data.val)
# predicates = predicate_data.get_base_predicates(logp)
# predicate_search = PredicateSearch(predicates)


# In[13]:


# predicate = predicate_search.search_features(features=data.drop('val',axis=1).columns, c=.7)


# In[14]:


# norm = NormAnomalyDetection(mean, var)


# In[15]:


# pred_predy = pred.predict(data)


# In[16]:


# norm_predy = norm.predict(data, c=.95)


# In[17]:


# norm_predy.mean()


# In[18]:


# pred_predy.mean()


# In[19]:


#     def predicate_labels(self, predicate, data):
#         label = data.index.isin(predicate.selected_index).astype(int)
#         return label
        
#     def evaluate_prediction(self, y, predy):
#         true_positives = true_positives = ((y == 1).astype(int) * (predy == 1).astype(int)).sum()
#         if predy.sum() > 0:
#             precision = true_positives / predy.sum()
#         else:
#             precision = 0
#         if y.sum() > 0:
#             recall = true_positives / y.sum()
#         else:
#             recall = 0
#         if precision + recall > 0:
#             f1 = 2 * (precision*recall) / (precision + recall)
#         else:
#             f1 = 0
#         return precision, recall, f1
    
#     def evalutate_data(self, data, pred_label):
#         att = data.drop('val', axis=1).columns
#         med_label = (np.all(data[att] >= self.med_outlier_lower, axis=1
#                           ) & np.all(data[att] <= self.med_outlier_upper, axis=1)).astype(int)
#         high_label = (np.all(data[att] >= self.high_outlier_lower, axis=1
#                           ) & np.all(data[att] <= self.high_outlier_upper, axis=1)).astype(int)
#         med_precision, med_recall, med_f1 = self.evaluate_prediction(med_label, pred_label)
#         high_precision, high_recall, high_f1 = self.evaluate_prediction(high_label, pred_label)
#         return med_precision, med_recall, med_f1, high_precision, high_recall, high_f1
    
#     def evaluate_predicate(self, predicate, data):
#         pred_label = self.predicate_labels(predicate, data)
#         return self.evalutate_data(data, pred_label)


# In[20]:


# def evaluate_data(predicate_search, synth, synth_data):
#     results = []
#     for c in np.linspace(0, 1, 50):
#         p = predicate_search.search_features(
#                       c=c, features=synth_data.drop('val', axis=1).columns, maxiters=10)
        
#         (med_precision, med_recall, med_f1,
#         high_precision, high_recall, high_f1) = synth.evaluate_predicate(p, synth_data)
#         results.append([c, med_precision, med_recall, med_f1, high_precision, high_recall, high_f1])
#     results = pd.DataFrame(results, columns=['c', 'med_precision', 'med_recall', 'med_f1', 'high_precision',
#                                              'high_recall', 'high_f1'])
#     return results


# In[21]:


# def evaluate_norm(synth, synth_data, mean, var):
#     results = []
#     for c in np.linspace(0, 1, 50):
#         m = NormAnomalyDetection(mean, var)
#         predy = m.predict(synth_data.val, c=c)
#         (med_precision, med_recall, med_f1,
#         high_precision, high_recall, high_f1) = synth.evalutate_data(synth_data, predy)
#         results.append([c, med_precision, med_recall, med_f1, high_precision, high_recall, high_f1])
#     results = pd.DataFrame(results, columns=['c', 'med_precision', 'med_recall', 'med_f1', 'high_precision',
#                                              'high_recall', 'high_f1'])
#     return results


# In[22]:


# def evaluate_synth(mu, m, mean=10, var=10, n_samples=100):
#     all_results = []
#     synth = SynthSingleTarget(mu=mu, m=m)
#     for i in range(n_samples):
#         data = synth.generate_data()
#         predicate_data = PredicateData(data)
#         disc_data = predicate_data.disc_data
#         logp = st.norm(mean, var).logpdf(data.val)
#         predicates = predicate_data.get_base_predicates(logp)
#         predicate_search = PredicateSearch(predicates)
#         results = evaluate_data(predicate_search, synth, data)
#         all_results.append(results)
#     all_results = pd.concat(all_results)
#     return all_results


# In[23]:


# synth = SynthSingleTarget(mu=80, m=2)
# data = synth.generate_data()
# predicate_data = PredicateData(data)
# disc_data = predicate_data.disc_data
# logp = st.norm(10, 10).logpdf(data.val)
# predicates = predicate_data.get_base_predicates(logp)
# predicate_search = PredicateSearch(predicates)


# In[24]:


# norm_results = evaluate_norm(synth, data, 10, 10)


# In[25]:


# pred_results = evaluate_data(predicate_search, synth, data)


# In[26]:


# pred_results.set_index('c').med_f1.plot()
# norm_results.set_index('c').med_f1.plot()


# In[27]:


# easy_2d = evaluate_synth(80, 2)
# easy_2d = pd.read_csv('easy_2d_results.csv')


# In[28]:


# sns.lineplot(x="c", y="med_precision", data=easy_2d)
# sns.lineplot(x="c", y="high_precision", data=easy_2d)


# In[29]:


# sns.lineplot(x="c", y="med_recall", data=easy_2d)
# sns.lineplot(x="c", y="high_recall", data=easy_2d)


# In[30]:


# sns.lineplot(x="c", y="med_f1", data=easy_2d)
# sns.lineplot(x="c", y="high_f1", data=easy_2d)


# In[31]:


# # hard_2d = evaluate_synth(40, 2)
# hard_2d = pd.read_csv('hard_2d_results.csv')


# In[32]:


# sns.lineplot(x="c", y="med_precision", data=hard_2d)
# sns.lineplot(x="c", y="high_precision", data=hard_2d)


# In[33]:


# sns.lineplot(x="c", y="med_recall", data=hard_2d)
# sns.lineplot(x="c", y="high_recall", data=hard_2d)


# In[34]:


# sns.lineplot(x="c", y="med_f1", data=hard_2d)
# sns.lineplot(x="c", y="high_f1", data=hard_2d)


# In[35]:


# sns.lineplot(x="c", y="med_precision", data=easy_3d)
# sns.lineplot(x="c", y="high_precision", data=easy_3d)


# In[36]:


# sns.lineplot(x="c", y="med_recall", data=easy_3d)
# sns.lineplot(x="c", y="high_recall", data=easy_3d)


# In[37]:


# sns.lineplot(x="c", y="med_f1", data=easy_3d)
# sns.lineplot(x="c", y="high_f1", data=easy_3d)


# In[38]:


# hard_3d = pd.read_csv('hard_3d_results.csv')


# In[39]:


# sns.lineplot(x="c", y="med_precision", data=hard_3d)
# sns.lineplot(x="c", y="high_precision", data=hard_3d)


# In[40]:


# sns.lineplot(x="c", y="med_recall", data=hard_3d)
# sns.lineplot(x="c", y="high_recall", data=hard_3d)


# In[41]:


# sns.lineplot(x="c", y="med_f1", data=hard_3d)
# sns.lineplot(x="c", y="high_f1", data=hard_3d)


# In[42]:


# synth = SynthSingleTarget(mu=80, m=4)
# data = synth.generate_data()
# predicate_data = PredicateData(data)
# disc_data = predicate_data.disc_data
# logp = st.norm(10, 10).logpdf(data.val)
# predicates = predicate_data.get_base_predicates(logp)
# predicate_search = PredicateSearch(predicates)


# In[43]:


# p = predicate_search.search_features(c=1, features=data.drop('val', axis=1).columns, maxiters=100)

