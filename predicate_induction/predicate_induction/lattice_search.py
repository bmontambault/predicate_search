import numpy as np

from .predicate_induction import PredicateInduction
from ..predicate.predicate import Conjunction
from ..hypothesis_testing.hypothesis_testing import ExpNormMix
from ..anomaly_detection.anomaly_detection import RobustCov

class LatticeSearch(PredicateInduction):

    def __init__(self, data, dtypes=None, targets=None, anomaly_model=RobustCov, slice_model=ExpNormMix, bins=100):
        super().__init__(data, dtypes, targets, anomaly_model, slice_model)
        self.bins = bins
        self.feature_base_predicates = self.get_feature_base_predicates()

    def insert_sorted(self, predicate, posterior_odds, priority_queue):
        if len(priority_queue) == 0 or posterior_odds <= priority_queue[-1]['posterior_odds']:
            priority_queue.append({'predicate': predicate, 'posterior_odds': posterior_odds})
        else:
            for i in range(len(priority_queue)):
                if posterior_odds >= priority_queue[i]['posterior_odds']:
                    priority_queue.insert(i, {'predicate': predicate, 'posterior_odds': posterior_odds})
                    break

    def initialize_predicates(self):
        remaining = []
        feature_base_predicate_odds = {}
        merged_feature_base_predicate_odds = {}
        for feature, base_predicates in self.feature_base_predicates.items():
            sorted_predicates = []
            feature_base_predicate_odds[feature] = []
            for predicate in base_predicates:
                conjunction = Conjunction({predicate.feature: predicate}, self.data)
                posterior_odds = self.get_log_posterior_odds(predicate)
                self.insert_sorted(conjunction, posterior_odds, sorted_predicates)
                feature_base_predicate_odds[feature].append({'predicate': predicate, 'posterior_odds': posterior_odds})
            sorted_predicates = self.merge_feature_predicates(sorted_predicates)
            predicate_posterior_odds = [{'predicate': list(p['predicate'].feature_predicate.values())[0],
                                         'posterior_odds': p['posterior_odds']} for p in sorted_predicates]
            merged_feature_base_predicate_odds[feature] = sorted(predicate_posterior_odds, key=lambda x: x['predicate'].values[-1])
            for sorted_predicate in sorted_predicates:
                self.insert_sorted(sorted_predicate['predicate'], sorted_predicate['posterior_odds'], remaining)
        remaining = [p for p in remaining if p['posterior_odds'] > 0]
        return remaining, feature_base_predicate_odds, merged_feature_base_predicate_odds

    def is_subsumed(self, predicate, posterior_odds, all_accepted):
        features = [feature for feature in all_accepted.keys() if feature == tuple([f for f in predicate.features if f in feature])]
        for feature in features:
            for p in all_accepted[feature]:
                if p['predicate'].contains(predicate) and p['posterior_odds'] >= posterior_odds:
                    return True
        return False

    def get_adjacent(self, predicate, feature, feature_base_predicate_odds):
        feature_predicates = feature_base_predicate_odds[feature]
        if predicate.feature_predicate[feature].values == feature_predicates[0]['predicate'].values:
            return feature_predicates[1:]
        elif predicate.feature_predicate[feature].values == feature_predicates[-1]['predicate'].values:
            return list(reversed(feature_predicates))[1:]

        left_adjacent = []
        right_adjacent = []
        left = False
        for i in range(len(feature_predicates)):
            p = feature_predicates[i]
            if not left and not p['predicate'].values[0] == predicate.feature_predicate[feature].values[0]:
                left_adjacent.insert(0, p)
            if p['predicate'].values[1] >= predicate.feature_predicate[feature].values[0] or p['predicate'].values[0] == predicate.feature_predicate[feature].values[0]:
                left = True
            if left and p['predicate'].values[0] >= predicate.feature_predicate[feature].values[1]:
                right_adjacent = feature_predicates[i:]
                break

        adjacent = []
        while len(left_adjacent) > 0 or len(right_adjacent) > 0:
            if len(left_adjacent) > 0:
                adjacent.append(left_adjacent.pop(0))
            if len(right_adjacent) > 0:
                adjacent.append(right_adjacent.pop(0))
        return adjacent

    def expand_predicate(self, predicate, posterior_odds, all_accepted, feature_base_predicate_odds):
        expanded = []
        for feature, predicates in feature_base_predicate_odds.items():
            if feature not in predicate.features:
                for p in predicates:
                    if predicate.overlaps(p['predicate']) > 1 and p['posterior_odds'] > 0:
                        new_predicate = predicate.merge(p['predicate'])
                        new_posterior_odds = self.get_log_posterior_odds(new_predicate)
                        if not self.is_subsumed(new_predicate, new_posterior_odds, all_accepted) and new_posterior_odds > posterior_odds:
                            self.insert_sorted(new_predicate, new_posterior_odds, expanded)
        return expanded

    def stretch_predicate(self, predicate, posterior_odds, feature_base_predicate_odds):
        for feature in predicate.features:
            if self.dtypes[feature] != 'discrete':
                adjacent = self.get_adjacent(predicate, feature, feature_base_predicate_odds)
                relax = True
            else:
                adjacent = feature_base_predicate_odds[feature]
                adjacent = sorted(adjacent, key=lambda x: x['posterior_odds'], reverse=True)
                relax = False
            predicate, posterior_odds, _ = self.merge_predicate_list(predicate, posterior_odds, adjacent, True, relax=relax)
        return predicate, posterior_odds

    def merge_feature_predicates(self, predicates, only_adjacent=True, relax=True):
        predicates = predicates.copy()
        merged_predicates = []
        while len(predicates) > 0:
            p = predicates.pop(0)
            p, posterior_odds, predicates = self.merge_predicate_list(p['predicate'], p['posterior_odds'], predicates, only_adjacent, relax)
            merged_predicates.append({'predicate': p, 'posterior_odds': posterior_odds})
        return merged_predicates

    def merge_predicate_list(self, predicate, posterior_odds, other_predicates, only_adjacent=True, relax=True):
        for i in range(len(other_predicates)):
            new_predicate = other_predicates[i]['predicate']
            if predicate.is_adjacent(new_predicate):
                merged_predicate = predicate.merge(new_predicate)
                merged_posterior_odds = self.get_log_posterior_odds(merged_predicate)
                if (relax and merged_posterior_odds > posterior_odds - 10**-4) or (not relax and merged_posterior_odds > posterior_odds):
                    del other_predicates[i]
                    return self.merge_predicate_list(merged_predicate, merged_posterior_odds, other_predicates, only_adjacent, relax)
        return predicate, posterior_odds, other_predicates

    def search(self, threshold=10, max_iters=1000):
        remaining, feature_base_posterior_odds, merged_feature_base_posterior_odds = self.initialize_predicates()
        all_accepted = {}
        i = 0
        while len(remaining) > 0 and i < max_iters:
            predicate_posterior_odds = remaining.pop(0)
            predicate = predicate_posterior_odds['predicate']
            posterior_odds = predicate_posterior_odds['posterior_odds']
            if not self.is_subsumed(predicate, posterior_odds, all_accepted):
                expanded = self.expand_predicate(predicate, posterior_odds, all_accepted, merged_feature_base_posterior_odds)
                for expanded_predicate in expanded:
                    self.insert_sorted(expanded_predicate['predicate'], expanded_predicate['posterior_odds'], remaining)
                if len(expanded) == 0 and posterior_odds > np.log(threshold):
                    if predicate.features in all_accepted:
                        all_accepted[predicate.features].append({'predicate': predicate, 'posterior_odds': posterior_odds})
                    else:
                        all_accepted[predicate.features] = [{'predicate': predicate, 'posterior_odds': posterior_odds}]
            i+= 1

        merged = {}
        for k, v in all_accepted.items():
            merged[k] = self.merge_feature_predicates(v, only_adjacent=False, relax=False)

        stretched = {}
        i = 0
        for k, v in merged.items():
            stretched[k] = []
            for vi in v:
                new_predicate, new_posterior_odds = self.stretch_predicate(vi['predicate'], vi['posterior_odds'], feature_base_posterior_odds)
                trimed_new_predicate = new_predicate.trim()
                if not self.is_subsumed(trimed_new_predicate, new_posterior_odds, stretched):
                    stretched[k].append({'predicate': trimed_new_predicate, 'posterior_odds': new_posterior_odds})
                i += 1
        return stretched