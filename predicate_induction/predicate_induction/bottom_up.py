import pandas as pd
import numpy as np
from heapq import merge

from .predicate_induction import PredicateInduction
from ..predicate.predicate import BasePredicate, Conjunction, Disjunction
from ..hypothesis_testing.hypothesis_testing import ExpNormMix
from ..anomaly_detection.anomaly_detection import RobustCov

class BottomUp(PredicateInduction):

    def __init__(self, data, dtypes, targets=None, anomaly_model=RobustCov, slice_model=ExpNormMix, bins=100):
        super().__init__(data, dtypes, targets, anomaly_model, slice_model)
        self.bins = bins
        self.feature_base_predicates, self.base_predicates = self.get_base_predicates()

    def sort_base_predicates(self, predicates):
        return sorted(predicates, key=lambda x: self.slice_model(self.scores, x.indices, x.size).log_bayes_factor(),
                      reverse=True)

    def get_base_predicates(self):
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
            feature_base_predicates[feature] = self.sort_base_predicates(predicates)

        base_predicates = [Conjunction({base_p.feature: base_p}, self.data) for base_p in
                            merge(*list(feature_base_predicates.values()),
                            key=lambda x: self.slice_model(self.scores,x.indices, x.size).log_bayes_factor(),
                            reverse=True)]
        return feature_base_predicates, base_predicates

    def merge_predicates(self, predicate, predicates):
        old_score = self.slice_model(self.scores, predicate.indices, predicate.size).log_bayes_factor()
        for i in range(len(predicates)):
            new_predicate = predicates[i]
            # print(predicate, new_predicate, predicate.is_adjacent(new_predicate))
            if predicate.is_adjacent(new_predicate):
                merged_predicate = predicate.merge(new_predicate)
                merged_score = self.slice_model(self.scores, merged_predicate.indices, merged_predicate.size).log_bayes_factor()
                # print(old_score, merged_score, merged_score >= old_score - 10 ** -10)
                if merged_score >= old_score - 10 ** -10:
                    del predicates[i]
                    return self.merge_predicates(merged_predicate, predicates)
        return predicate

    def merge_feature(self, predicate, feature):
        base_predicates = self.feature_base_predicates[feature].copy()
        predicate = self.merge_predicates(predicate, base_predicates)
        return predicate

    def intersect_feature(self, predicate, feature):
        best_score = -np.inf
        best_predicate = None
        for new_predicate in self.feature_base_predicates[feature]:
            merged_predicate = predicate.merge(new_predicate)
            merged_score = self.slice_model(self.scores, merged_predicate.indices,
                                            merged_predicate.size).log_bayes_factor()
            if merged_score > best_score:
                best_score = merged_score
                best_predicate = merged_predicate
        return best_predicate, best_score

    def intersect_merge(self, predicate):
        intersected_predicates = [[feature] + list(self.intersect_feature(predicate, feature)) for feature in
                                  self.feature_base_predicates.keys() if feature not in
                                  predicate.feature_predicate.keys()]
        sorted_intersected_predicates = [p[:2] for p in sorted(intersected_predicates, key=lambda x: x[2], reverse=True)]
        #print(sorted_intersected_predicates)
        for feature, new_predicate in sorted_intersected_predicates:
            old_score = self.slice_model(self.scores, predicate.indices, predicate.size).log_bayes_factor()
            merged_feature = self.merge_feature(new_predicate, feature)
            merged_predicate = predicate.merge(merged_feature.feature_predicate[feature])
            merged_score = self.slice_model(self.scores, merged_predicate.indices,
                                            merged_predicate.size).log_bayes_factor()
            if merged_score > old_score:
                predicate = merged_predicate
        return predicate

    def merge(self, predicate):
        conjunction = self.merge_feature(predicate, predicate.features[0])
        conjunction = self.intersect_merge(conjunction)
        return conjunction

    def search(self, threshold=10):
        accepted = Disjunction([], self.data)
        rejected = Disjunction([], self.data)
        for conjunction in self.base_predicates:
            # print(conjunction, not accepted.contains(conjunction))
            # if not (accepted.contains(conjunction)):# or rejected.contains(conjunction)):
            conjunction = self.merge(conjunction)
            log_posterior_odds = self.slice_model(self.scores, conjunction.indices,
                                            conjunction.size).log_posterior_odds()
            posterior_odds = np.exp(log_posterior_odds)
            print(conjunction, posterior_odds)
            print('______')
            if posterior_odds >= threshold:
                accepted = accepted.merge(conjunction)
            # else:
            #     rejected = rejected.merge(conjunction)
        return accepted