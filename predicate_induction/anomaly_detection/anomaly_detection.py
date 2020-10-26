import scipy.stats as st
import pandas as pd
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import pymc3 as pm

class AnomalyDetection(object):

    def format_dtypes(self, data, dtypes=None, continuous_encoding=None, discrete_encoding=None, bins=None):
        if dtypes is None:
            return data
        else:
            dtype_data = pd.DataFrame()
            for col in data.columns:
                if col in dtypes:
                    if dtypes[col] == 'continuous':
                        if continuous_encoding == 'bin':
                            if bins is None:
                                step = ((data[col].max() - data[col].min()) / self.bins)
                                bins = np.arange(data[col].min(), data[col].max() + step, step)
                            dtype_data[col] = pd.cut(data[col], bins=bins)
                    if dtypes[col] == 'date':
                        dtype_data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
                    elif dtypes[col] == 'discrete':
                        if discrete_encoding == 'one_hot':
                            for new_col in sorted(data[col].unique()):
                                dtype_data[f'{col}_{new_col}'] = (data[col] == new_col).astype(int)
                    else:
                        dtype_data[col] = data[col]
                else:
                    dtype_data[col] = data[col]
            return dtype_data

    def base_fit(self, data):
        raise NotImplemented

    def fit(self, data, dtypes=None, **format_args):
        data = self.format_dtypes(data, dtypes, self.continuous_encoding, self.discrete_encoding, **format_args)
        self.base_fit(data)

    def base_score(self, data):
        raise NotImplemented

    def score(self, data, dtypes=None):
        data = self.format_dtypes(data, dtypes, self.continuous_encoding, self.discrete_encoding)
        base_score = self.base_score(data)
        return base_score
        score = base_score - base_score.min()
        normalized_score = score / (score.max() - score.min())
        return normalized_score

class Multinomial(AnomalyDetection):

    continuous_encoding = 'bins'
    discrete_encoding = None

    def __init__(self, nbins=100):
        self.nbins = nbins

    def base_fit(self, data):
        self.counts = {col: data[col].value_counts() for col in data.columns}

    def score_col(self, data, col, alpha=1):
        counts = data[col].map(self.counts[col]).fillna(0) + alpha
        logp = np.log(counts / counts.sum()).values
        return -logp

    def base_score(self, data):
        feature_scores = np.array([self.score_col(data, col) for col in data.columns])
        return feature_scores.sum(axis=0)

class Normal(AnomalyDetection):

    continuous_encoding = None
    discrete_encoding = 'one_hot'

    def base_fit(self, data):
        self.mean = data.mean()
        self.std = data.std()

    def base_score(self, data):
        x = data.values
        logp = st.norm(self.mean, self.std).logpdf(x).ravel()
        return -logp

class BayesNormal(AnomalyDetection):

    continuous_encoding = None
    discrete_encoding = 'one_hot'

    def __init__(self, prior_mean, prior_var, alpha, beta):
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.alpha = alpha
        self.beta = beta

    def add(self, x):
        new_prior_var = self.prior_var + 1
        self.alpha += .5
        self.beta += (self.prior_var / new_prior_var) * ((x - self.prior_mean) ** 2 / 2.)
        self.prior_mean = (self.prior_var * self.prior_mean + x) / new_prior_var
        self.prior_var = new_prior_var

    def remove(self, x):
        new_prior_var = self.prior_var - 1
        self.alpha -= .5
        self.beta -= (self.prior_var / new_prior_var) * ((x - self.prior_mean) ** 2 / 2.)
        self.prior_mean = (self.prior_var * self.prior_mean - x) / new_prior_var
        self.prior_var = new_prior_var

    def logpdf(self, x):
        df = 2 * self.alpha
        scale = self.beta * (self.prior_var + 1) / (self.prior_var * self.alpha)
        return st.t(df=df, loc=self.prior_mean, scale=scale).logpdf(x)

    def base_fit(self, data):
        n = len(data)
        mean = data.mean()
        se = .5*((data - mean)**2).sum()

        new_prior_var = self.prior_var + n
        self.alpha += .5*n
        self.beta += se + (self.prior_var*n / new_prior_var) * ((mean - self.prior_mean) ** 2 / 2.)
        self.prior_mean = (self.prior_var*self.prior_mean + mean*n) / new_prior_var
        self.prior_var = new_prior_var

    def base_score(self, data):
        return -self.logpdf(data)

class RobustCov(AnomalyDetection):

    continuous_encoding = None
    discrete_encoding = 'one_hot'

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
        if data.shape[1] > 1:
            cov = MinCovDet().fit(data).covariance_
        else:
            cov = data.var().values[None]
        mean = data.median()
        return mean, cov

    def base_fit(self, data):
        if self.how == 'mcd':
            mean, cov = self.fit_mcd(data)
        elif self.how == 't':
            mean, cov = self.fit_t(data)
        self.mean = pd.Series(mean, index=data.columns)
        self.cov = pd.DataFrame(cov, index=data.columns, columns=data.columns)

    def base_score(self, data):
        m = self.mean[data.columns].values
        c = self.cov[data.columns].loc[data.columns].values
        x = data.values
        logp = st.multivariate_normal(m, c, allow_singular=True).logpdf(x)
        return -logp

class LOF(AnomalyDetection):

    continuous_encoding = None
    discrete_encoding = 'one_hot'

    def base_fit(self, data):
        self.model = LocalOutlierFactor()
        self.model.fit(data)

    def base_score(self, data):
        score = -self.model.negative_outlier_factor_
        return score

class IsoForest(AnomalyDetection):

    continuous_encoding = None
    discrete_encoding = 'one_hot'

    def base_fit(self, data):
        self.model = IsolationForest()
        self.model.fit(data)

    def base_score(self, data):
        score = -self.model.score_samples(data)
        return score