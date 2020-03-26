import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns

from .predicate_data import PredicateData
from .predicate import CompoundPredicate

class NormData:

    def __init__(self, n, m, predicate_n, predicate_m, targets=None, alpha=1, beta=10, q=8, bins=100):
        self.n = n
        self.m = m
        self.predicate_n = predicate_n
        self.predicate_m = predicate_m
        if targets is None:
            self.targets = [f'f{i}' for i in range(m)]
        else:
            self.targets = targets
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.bins = bins

        self.clean, self.clean_mean, self.clean_cov = self.generate_norm(n, m, alpha, beta)
        self.anomalies, self.anom_mean, self.anom_cov = self.generate_norm(n, m, alpha, beta)
        self.predicate_clean = PredicateData(self.clean)
        self.predicate_anomalies = PredicateData(self.anomalies)
        self.predicate = self.generate_predicate(self.predicate_m, self.predicate_n)
        self.tainted = self.insert_anomalies(self.targets, self.predicate)
        # self.disc_predicate, self.predicate, self.y = self.insert_anomalies()

    def generate_norm(self, n, m, alpha=1, beta=.1):
        mean = np.random.normal(0, 10, size=m)
        cov = st.invwishart(df=alpha + m, scale=np.ones(m) * beta).rvs()
        if m > 1:
            data = np.random.multivariate_normal(mean, cov, size=n)
        else:
            data = np.random.normal(mean, cov, size=n)
        df = pd.DataFrame(data)
        df.columns = [f"f{col}" for col in df.columns]
        return df, mean, cov

    def generate_feature_predicate(self, feature, n=5):
        predicates = [p for p in self.predicate_clean.get_base_predicates() if p.feature == feature]
        p0 = predicates.pop(np.random.choice(np.arange(len(predicates))))
        for i in range(n-1):
            adj = [i for i in range(len(predicates)) if predicates[i].is_adjacent(p0)]
            new_p = predicates.pop(np.random.choice(adj))
            p0 = p0.merge(new_p)
        return p0

    def generate_predicate(self, m=2, n=5, maxiters=100):
        for i in range(maxiters):
            features = np.random.choice(self.predicate_clean.features, size=m)
            base_predicates = [self.generate_feature_predicate(feature, n=n) for feature in features]
            predicate = CompoundPredicate(base_predicates)
            if len(predicate.selected_index) > 0:
                return predicate

    def insert_anomalies(self, targets, predicate):
        if targets is None:
            targets = self.predicate_clean.features
        clean_target = self.clean[~self.clean.index.isin(predicate.selected_index)][targets]
        anomalies_target = self.anomalies[self.anomalies.index.isin(predicate.selected_index)][targets]
        other_columns = self.clean[[col for col in self.clean.columns if col not in targets]]
        tainted = pd.concat([other_columns, pd.concat([clean_target, anomalies_target])], axis=1).reset_index(drop=True)
        return tainted

    def plot(self):
        concatenated = pd.concat([self.clean.assign(label='data'),
                                  self.anomalies.assign(label='anomaly')])
        sns.pairplot(concatenated, hue='label')

    def plot2d(self, x='f0', y='f1'):
        sns.scatterplot(x=x, y=y, data=self.tainted.assign(label=self.y), hue='label')
