import scipy.stats as st
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import pymc3 as pm

class Model(object):

    def fit(self, data):
        raise NotImplemented

    def score(self, data):
        raise NotImplemented

class RobustCov(Model):

    def __init__(self, how='mcd', nu=5):
        self.how = how
        self.nu = nu

    def fit_t(self, data):
        with pm.Model() as model:
            packed_L = pm.LKJCholeskyCov('packed_L', n=data.shape[1], eta=2., sd_dist=pm.HalfCauchy.dist(2.5))
            L = pm.expand_packed_triangular(data.shape[1], packed_L)
            cov = pm.Deterministic('cov', L.dot(L.T))
            mean = pm.Normal('mean', mu=0, sigma=10, shape=data.shape[1])
            obs = pm.MvStudentT('obs', nu=self.nu, mu=mean, chol=L, observed=data)
        params = pm.find_MAP(model=model, progressbar=False)
        return params['mean'], params['cov']

    def fit_mcd(self, data):
        cov = MinCovDet().fit(data).covariance_
        mean = data.median()
        return mean, cov

    def fit(self, data):
        if self.how == 'mcd':
            mean, cov = self.fit_mcd(data)
        elif self.how == 't':
            mean, cov = self.fit_t(data)
        self.mean = pd.Series(mean, index=data.columns)
        self.cov = pd.DataFrame(cov, index=data.columns, columns=data.columns)

    def score(self, data):
        m = self.mean[data.columns].values
        c = self.cov[data.columns].loc[data.columns].values
        x = data.values
        logp = st.multivariate_normal(m, c, allow_singular=True).logpdf(x)
        return -logp/1000

class LOF(Model):

    def fit(self, data):
        self.model = LocalOutlierFactor()
        self.model.fit(data)

    def score(self, data):
        score = -self.model.negative_outlier_factor_
        return score

class IsoForest(Model):

    def fit(self, data):
        self.model = IsolationForest()
        self.model.fit(data)

    def score(self, data):
        score = -self.model.score_samples(data)
        return score