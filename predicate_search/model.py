import pymc3 as pm
import scipy.stats as st
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

class Model(object):

    def fit(self, data):
        raise NotImplemented

    def logp(self, data):
        raise NotImplemented

    def distance(self, data):
        raise NotImplemented

class RobustNormal(Model):

    def fit_t(self, data, nu=5):
        with pm.Model() as model:
            packed_L = pm.LKJCholeskyCov('packed_L', n=data.shape[1], eta=2., sd_dist=pm.HalfCauchy.dist(2.5))
            L = pm.expand_packed_triangular(data.shape[1], packed_L)
            cov = pm.Deterministic('cov', L.dot(L.T))
            mean = pm.Normal('mean', mu=0, sigma=10, shape=data.shape[1])
            obs = pm.MvStudentT('obs', nu=nu, mu=mean, chol=L, observed=data)
        params = pm.find_MAP(model=model, progressbar=False)
        return params['mean'], params['cov']

    def fit_mcd(self, data):
        cov = MinCovDet().fit(data).covariance_
        mean = data.median()
        return mean, cov

    def fit(self, data, how='mcd'):
        if how == 'mcd':
            mean, cov = self.fit_mcd(data)
        elif how == 't':
            mean, cov = self.fit_t(data)
        self.mean = pd.Series(mean, index=data.columns)
        self.cov = pd.DataFrame(cov, index=data.columns, columns=data.columns)

    def logp(self, data):
        m = self.mean[data.columns].values
        c = self.cov[data.columns].loc[data.columns].values
        x = data.values
        logp = st.multivariate_normal(m, c).logpdf(x)
        return logp

    def distance(self, data):
        m = self.mean[data.columns].values
        c = self.cov[data.columns].loc[data.columns].values
        x = data.values
        VI = np.linalg.inv(c)
        delta = x - m
        dist = np.sqrt(np.einsum('nj,jk,nk->n', delta, VI, delta))
        return dist