import numpy as np
import scipy.stats as st

from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
BayesFactor=importr('BayesFactor', suppress_messages=True)

class NormalDist:

    def __init__(self, prior_mean, prior_observations, alpha, beta):
        self.prior_mean = prior_mean
        self.prior_observations = prior_observations
        self.alpha = alpha
        self.beta = beta

        self.df = 2 * self.alpha
        self.scale = (self.beta * (self.prior_observations + 1)) / (self.prior_observations * self.alpha)

    def marginal_likelihood(self, data):
        return st.t(loc=self.prior_mean, scale=self.scale, df=self.df).pdf(data)

    def log_marginal_likelihood(self, data):
        return st.t(loc=self.prior_mean, scale=self.scale, df=self.df).logpdf(data)

    def interval(self, alpha):
        return st.t(loc=self.prior_mean, scale=self.scale, df=self.df).interval(alpha)

class ExponentialDist:

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def marginal_likelihood(self, data):
        return st.lomax(scale=self.alpha, c=self.beta).pdf(data)

    def log_marginal_likelihood(self, data):
        return st.lomax(scale=self.alpha, c=self.beta).logpdf(data)

    def interval(self, alpha):
        return st.lomax(scale=self.alpha, c=self.beta).interval(alpha)

class SliceModel(object):

    def __init__(self, scores, slice_indices, size=1, threshold=10, prior_scale=1):
        self.scores = scores
        self.slice_indices = slice_indices
        self.size = size
        self.compliment_indices = [i for i in range(len(self.scores)) if i not in self.slice_indices]
        self.slice_size = len(self.slice_indices)
        self.compliment_size = len(self.compliment_indices)
        self.data_size = len(scores)
        if len(self.slice_indices) > 0:
            self.slice_scores = self.scores[self.slice_indices]
        else:
            self.slice_scores = np.array([])
        if len(self.compliment_indices) > 0:
            self.compliment_scores = self.scores[self.compliment_indices]
        else:
            self.compliment_scores = np.array([])
        self.threshold = threshold
        self.prior_scale = prior_scale

    def log_prior_odds(self, prior='predicate_size'):
        if prior == 'predicate_size':
            scale = self.size
        elif prior == 'slice_size':
            scale = self.slice_size
        elif prior == 'data_size':
            scale = self.data_size
        elif prior is None:
            return 0
        return np.log(1. / scale ** self.prior_scale)

    def log_bayes_factor(self):
        raise NotImplementedError

    def log_posterior_odds(self, prior='predicate_size'):
        return self.log_prior_odds(prior) + self.log_bayes_factor()

    def test(self):
        return np.exp(self.log_posterior_odds()) > self.threshold

class ExpNormMix(SliceModel):

    def __init__(self, scores, slice_indices, size=1, threshold=10, prior_scale=1, prior_mean=.5, prior_observations=6,
                 norm_alpha=7, norm_beta=1, exp_alpha=1, exp_beta=7.6):
        super().__init__(scores, slice_indices, size, threshold, prior_scale)
        self.normal_dist = NormalDist(prior_mean, prior_observations, norm_alpha, norm_beta)
        self.exponential_dist = ExponentialDist(exp_alpha, exp_beta)

    def log_bayes_factor(self):
        h0 = self.exponential_dist.log_marginal_likelihood(self.scores).sum()
        h1 = (self.normal_dist.log_marginal_likelihood(self.slice_scores).sum() +
              self.exponential_dist.log_marginal_likelihood(self.compliment_scores).sum())
        return h1 - h0

class JZS(SliceModel):

    def __init__(self, scores, slice_indices, size=1, threshold=10, prior_scale=1):
        super().__init__(scores, slice_indices, size, threshold, prior_scale)

    def log_bayes_factor(self):
        if len(self.slice_scores) <= 1 or len(self.compliment_scores) <= 1:
            return 1
        if self.slice_scores.mean() < self.compliment_scores.mean():
            return -np.inf
        else:
            res = BayesFactor.ttestBF(x=self.slice_scores, y=self.compliment_scores)
            bf = res.slots['bayesFactor'][0][0]
            return bf

# class JZS(SliceModel):
#
#     def __init__(self, scores, slice_indices, size=1, threshold=10, prior_scale=1):
#         super().__init__(scores, slice_indices, size, threshold, prior_scale)
#
#     def log_bayes_factor(self):
#         n = self.compliment_size * self.slice_size / (self.compliment_size + self.slice_size)
#         t = st.ttest_ind(self.slice_scores, self.compliment_scores, equal_var=False).statistic
#         return np.log(t**2)
#         return np.sqrt(n) * (1 + (t**2/(n-1))) ** (-n/2.)

    # def log_bayes_factor(self):
    #     n = self.compliment_size*self.slice_size/(self.compliment_size + self.slice_size)
    #     t_test = st.ttest_ind(self.slice_scores, self.compliment_scores, equal_var=False)
    #     t = t_test.statistic
    #
    #     bic0_bic1 = n * np.log(1 + (t**2/(n - 1))) - np.log(n)
    #     log_bayes_factor = (bic0_bic1/2.)
    #     return log_bayes_factor
