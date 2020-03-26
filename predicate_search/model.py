import pymc3 as pm
import scipy.stats as st
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

class RobustNormal:

    def __init__(self, nu=5):
        self.nu = nu

    def fit_t(self, data, nu=None):
        if nu is None:
            nu = self.nu
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

    def fit(self, data, how='mcd'):
        self.features = data.columns
        if how == 'mcd':
            mean, cov = self.fit_mcd(data)
        elif how == 't':
            mean, cov = self.fit_t(data)
        self.mean = pd.Series(mean, index=self.features)
        self.cov = pd.DataFrame(cov, index=self.features, columns=self.features)

    def score(self, data, targets=None):
        if targets is None:
            return st.multivariate_normal(self.mean, self.cov).logpdf(data)
        else:
            m = self.mean[targets].values
            c = self.cov[targets].loc[targets].values
            x = data[targets].values
            return st.multivariate_normal(m, c).logpdf(x)

    def get_distances(self, data, features=None):
        if features is None:
            features = self.features
        m = self.mean[features].values
        c = self.cov[features].loc[features].values
        x = data[features].values

        VI = np.linalg.inv(c)
        delta = x - m
        dist = np.sqrt(np.einsum('nj,jk,nk->n', delta, VI, delta))
        return dist

def det_dot(a, b):
    return (a * b[None, :]).sum(axis=-1)

class NormalModel:

    def fit(self, X, y):
        self.X = X
        self.y = y
        model = self.model()
        self.params = pm.find_MAP(model=model)
        self.forward_params = {k: v for k, v in self.params.items() if k != 'sigma' and 'log__' not in k}
        self.sigma = self.params['sigma']

    def predict(self, X):
        predy = self.forward(X, **self.forward_params)
        return predy

    def score(self, X, y):
        predy = self.predict(X)

        return st.norm(predy, self.sigma).logpdf(y)[0]

class Linear(NormalModel):

    def __init__(self, noise='normal', nu=5, alpha=10, beta=10):
        self.noise = noise
        self.nu = nu
        self.alpha = alpha
        self.beta = beta

    def model(self):
        with pm.Model() as model:
            bias = pm.Normal('bias', mu=0, sigma=self.alpha)
            weights = pm.Normal('weights', mu=0, sigma=self.beta, shape=self.X.shape[1])
            predy = self.forward(self.X, bias, weights)

            sigma = pm.HalfCauchy('sigma', 10, testval=.1)
            if self.noise == 'normal':
                obs = pm.Normal('obs', mu=predy, sigma=sigma, observed=self.y)
            elif self.noise == 'robust':
                obs = pm.StudentT('obs', nu=self.nu, mu=predy, sigma=sigma, observed=self.y)
        return model

    def forward(self, X, bias, weights):
        return det_dot(weights, X) + bias