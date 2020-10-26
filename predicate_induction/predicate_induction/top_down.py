import pandas as pd
import numpy as np

from .predicate_induction import PredicateInduction
from ..predicate.predicate import Predicate, Disjunction
from ..hypothesis_testing.hypothesis_testing import ExpNormMix
from ..anomaly_detection.anomaly_detection import RobustCov

class Node:

    def __init__(self, features, feature_binned_values, data, dtypes, slice_model, scores):
        self.features = features
        self.feature_binned_values = feature_binned_values
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
            binned_values = self.feature_binned_values[feature]
            if self.dtypes[feature] == 'discrete':
                values = binned_values
            elif self.dtypes[feature] in ['continuous', 'date']:
                values = (binned_values[0][0], binned_values[-1][-1])
            feature_values[feature] = values
        return Predicate([feature_values], self.data, self.dtypes)

    def split_node_feature(self, split_feature, index):
        left_feature_binned_values = {}
        right_feature_binned_values = {}
        for feature, binned_values in self.feature_binned_values.items():
            if feature != split_feature:
                left_feature_binned_values[feature] = binned_values
                right_feature_binned_values[feature] = binned_values
            else:
                left_feature_binned_values[feature] = binned_values[:index]
                right_feature_binned_values[feature] = binned_values[index:]

        features = list(set(self.features + [split_feature]))
        left_node = Node(features, left_feature_binned_values, self.data, self.dtypes, self.slice_model, self.scores)
        right_node = Node(features, right_feature_binned_values, self.data, self.dtypes, self.slice_model, self.scores)
        return left_node, right_node

    def split_node_feature_all(self, split_feature):
        return [self.split_node_feature(split_feature, i) for i in range(1, len(self.feature_binned_values[split_feature])-1)]

    def split_node(self):
        return [a for b in [self.split_node_feature_all(split_feature) for split_feature in self.feature_binned_values.keys()] for a in b]

class TopDown(PredicateInduction):

    def __init__(self, data, dtypes, targets=None, anomaly_model=RobustCov, slice_model=ExpNormMix, bins=100):
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

    def split_node(self, node):
        children = [max(child, key=lambda x: x.score) for child in node.split_node()]
        improved_children = [child for child in children if child.score > node.score]
        return improved_children

    def expand_node(self, node):
        children = self.split_node(node)
        if len(children) > 0:
            best_child = max(children, key=lambda x: x.score)
            best_score = best_child.score
            if best_score > node.score:
                return self.expand_node(best_child)
        return node

    def search(self, threshold=10):
        root = Node([], self.feature_binned_values, self.data, self.dtypes, self.slice_model, self.scores)
        root_children = sorted(self.split_node(root), key=lambda x: x.score, reverse=True)
        accepted = Disjunction([], self.data)
        for node in root_children:
            expanded_node = self.expand_node(node)
            log_posterior_odds = expanded_node.score
            posterior_odds = np.exp(log_posterior_odds)
            if posterior_odds >= threshold:
                conjunction = expanded_node.to_predicate().base_predicates[0]
                if not accepted.contains(conjunction):
                    accepted = accepted.merge(conjunction)
        return accepted
