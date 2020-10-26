import pandas as pd
import numpy as np

from ..hypothesis_testing.hypothesis_testing import ExpNormMix
from ..anomaly_detection.anomaly_detection import RobustCov
from ..predicate.predicate import BasePredicate

class PredicateInduction(object):

    def __init__(self, data, dtypes=None, targets=None, anomaly_model=RobustCov, slice_model=ExpNormMix):
        self.data = data
        if dtypes is None:
            self.dtypes = {col: 'continuous' for col in data.columns}
        else:
            self.dtypes = dtypes
        if targets is None:
            self.targets = self.data.columns
        else:
            self.targets = targets
        self.anomaly_model_class = anomaly_model
        self.slice_model = slice_model
        self.scores = self.get_scores()

    def get_scores(self):
        self.anomaly_model = self.anomaly_model_class()
        self.anomaly_model.fit(self.data[self.targets], self.dtypes)
        return self.anomaly_model.score(self.data[self.targets], self.dtypes)

    def get_feature_base_predicates(self):
        feature_base_predicates = {}
        for feature, dtype in self.dtypes.items():
            if dtype == 'discrete':
                values = [[value] for value in self.data[feature].unique().tolist()]
            elif dtype in ['continuous', 'date']:
                bins = pd.cut(self.data[feature], bins=self.bins, right=True)
                values = [(bin.left, bin.right) for bin in bins.sort_values().unique()]
                missing_values = []
                for i in range(len(values) - 1):
                    if values[i][1] != values[i+1][0]:
                        missing_values.append((values[i][1], values[i+1][0]))
                values = sorted(values + missing_values)
            else:
                values = []
            predicates = [BasePredicate(feature, value, self.data, dtype) for value in values]
            feature_base_predicates[feature] = predicates
        return feature_base_predicates

    def get_log_posterior_odds(self, predicate):
        log_posterior_odds = self.slice_model(self.scores, predicate.indices, predicate.size).log_posterior_odds(prior=None)
        return log_posterior_odds
        return np.exp(log_posterior_odds)