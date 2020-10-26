import pandas as pd
import numpy as np
from heapq import merge

from .predicate_induction import PredicateInduction
from ..predicate.predicate import BasePredicate, Conjunction, Disjunction, Predicate
from ..hypothesis_testing.hypothesis_testing import ExpNormMix
from ..anomaly_detection.anomaly_detection import RobustCov

class SearchNode:

    def __init__(self, feature_values, feature_disc_values, data, dtypes, slice_model, scores):
        self.feature_values = feature_values
        self.features = list(self.feature_values.keys())
        self.feature_disc_values = feature_disc_values
        self.data = data
        self.dtypes = dtypes
        self.slice_model = slice_model
        self.scores = scores
        self.score = self.get_score()

    def get_score(self):
        if len(self.features) > 0:
            predicate = self.to_predicate()
            return self.slice_model(self.scores, predicate.indices, predicate.size).log_posterior_odds()
        else:
            return 0

    def to_predicate(self):
        feature_values = {}
        for feature in self.features:
            if self.dtypes[feature] == 'discrete':
                values = [self.feature_binned_values[feature][i] for i in range(len(self.feature_values[feature]))
                          if self.feature_values[feature][i] == 1]
            elif self.dtypes[feature] in ['continuous', 'date']:
                values = tuple(self.feature_values[feature])
            feature_values[feature] = values
        return Predicate([feature_values], self.data, self.dtypes)

class ActiveSearch(PredicateInduction):

    def __init__(self, data, dtypes, targets=None, anomaly_model=RobustCov, slice_model=ExpNormMix, bins=10):
        super().__init__(data, dtypes, targets, anomaly_model, slice_model)
        self.bins = bins
        self.feature_binned_values = self.get_binned_values()

    def get_binned_values(self):
        feature_binned_values = {}
        for feature, dtype in self.dtypes.items():
            if dtype == 'discrete':
                values = [value for value in self.data[feature].unique().tolist()]
            elif dtype in ['continuous', 'date']:
                bins = pd.cut(self.data[feature], bins=self.bins, right=True)
                values = [(bin.left, bin.right) for bin in bins.sort_values().unique()]
            feature_binned_values[feature] = values
        return feature_binned_values

    def vectorize_values(self, feature, values):
        binned_values = self.feature_binned_values[feature]
        if self.dtypes[feature] == 'discrete':
            return [binned_values[i] in values for i in range(len(binned_values))]
        elif self.dtypes[feature] in ['continuous', 'date']:
            return list(values)

    # def get_nodes(self):
    #     nodes = []
    #     for feature, binned_values in self.feature_binned_values.items():
    #         feature_values = {}
    #
    #
    #         if self.dtypes[feature] == 'discrete':
    #             for i in range(len(binned_values)):
    #
    #
    #             values = [[val] for val in self.feature_binned_values[feature]]
    #         elif self.dtypes[feature] in ['continuous', 'date']:
    #             values = self.feature_binned_values[feature]
    #
    #
    #     return [SearchNode(feature_values, self.feature_disc_values, self.data, self.dtypes, self.slice_model,
    #                        self.scores) for feature_values in feature_values_list]

    def search(self, threshold=10):
        nodes = self.get_nodes()