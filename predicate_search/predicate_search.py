import itertools
import numpy as np
import copy

from .predicate import ContBasePredicate, DiscBasePredicate, CompoundPredicate
from .predicate_data import PredicateData

class PredicateSearch:

    def __init__(self, data, logp, disc_cols, c=1, b=.1, quantile=.25):
        self.data = data
        self.logp = logp
        self.predicate_data = PredicateData(data, disc_cols=disc_cols)
        self.set_predicates()

        self.c = c
        self.b = b
        self.quantile = quantile

    def set_predicates(self, index=None):
        self.predicates = self.predicate_data.get_base_predicates(self.logp, index)
        self.all_features = list(set([p.feature for p in self.predicates]))
        self.predicate_map = {(self.predicates[i].feature, np.min(self.predicates[i].values)): i for i in range(len(self.predicates))}

    def get_base_predicate(self, feature, value):
        if (feature, value) in self.predicate_map.keys():
            return self.predicates[self.predicate_map[(feature, value)]]
        else:
            if feature in self.predicate_data.disc_cols:
                return DiscBasePredicate(feature, [value], [], self.logp, None)
            else:
                return ContBasePredicate(feature, [(value, value)], [], self.logp, None)

    def get_predicate(self, feature, values):
        val1 = values[0]
        if type(val1) == int:
            predicate = self.get_base_predicate(feature, val1)
            for i in range(1, len(values)):
                predicate = predicate.merge(self.get_base_predicate(feature, values[i]))
        else:
            predicate = self.get_base_predicate(feature, val1[0])
            for val in range(val1[0]+1, val1[1]+1):
                predicate = predicate.merge(self.get_base_predicate(feature, val))
            for i in range(1, len(values)):
                for val in range(values[i][0], values[i][1] + 1):
                    predicate = predicate.merge(self.get_base_predicate(feature, val))
        return predicate

    def group_predicates(self, predicates, c):
        grouped = {tuple(k): list(g) for k, g in itertools.groupby(predicates, key=lambda x: x.features)}
        for k, v in grouped.items():
            grouped[k] = sorted(v, key=lambda x: x.get_influence(c), reverse=True)
        sorted_features = sorted(grouped.keys(), key=lambda k: grouped[k][0].get_influence(c), reverse=True)
        return sorted_features, grouped

    def merge_same_features(self, predicates, c):
        predicates = predicates.copy()
        merged_predicates = []
        while len(predicates) > 0:
            p = predicates.pop(0)
            p, predicates = self.merge_predicate_list(p, predicates, c)
            merged_predicates.append(p)
        return merged_predicates

    def merge_predicate_list(self, p, predicates, c):
        old_influence = p.get_influence(c)
        for i in range(len(predicates)):
            new_p = predicates[i]
            if p.is_adjacent(new_p):
                merged_p = p.merge(new_p)
                new_influence = merged_p.get_influence(c)
                # print(p, old_influence, merged_p, new_influence)
                if new_influence >= old_influence:
                    del predicates[i]
                    return self.merge_predicate_list(merged_p, predicates, c)
        return p, predicates

    def merge_adjacent(self, predicates, c):
        merged = []
        sorted_features, grouped = self.group_predicates(predicates, c)
        for k in sorted_features:
            merged_features = self.merge_same_features(grouped[k], c)
            merged += merged_features
        merged_filtered = ([next(v) for k, v in itertools.groupby(merged, lambda x: x.query)])
        return merged_filtered

    def get_base_missing_intervals(self, predicate):
        missing_intervals = []
        for i in range(len(predicate.values) - 1):
            interval = None
            left_val = predicate.values[i][1] + 1
            right_val = predicate.values[i + 1][0] - 1
            for v in range(left_val, right_val + 1):
                new_p = self.get_base_predicate(predicate.feature, v)
                if interval is None:
                    interval = new_p
                else:
                    interval = interval.merge(new_p)
                if interval is not None:
                    missing_intervals.append(interval)

        if len(missing_intervals) > 0:
            interval_predicate = missing_intervals[0]
            for p in missing_intervals[1:]:
                interval_predicate = interval_predicate.merge(p)
            return interval_predicate
        else:
            return None

    def get_missing_intervals(self, predicate):
        if type(predicate) == CompoundPredicate:
            missing_intervals = [self.get_base_missing_intervals(p) for p in predicate.base_predicates]
        else:
            missing_intervals = [self.get_base_missing_intervals(predicate)]
        return [p for p in missing_intervals if p is not None]

    def merge_predicate(self, predicate, c, b):
        if type(predicate) == DiscBasePredicate:
            return predicate
        else:
            missing_intervals = self.get_missing_intervals(predicate)
            for interval in missing_intervals:
                old_prior = 1. / len(predicate.values) ** b
                new_p = predicate.merge(interval)
                new_prior = 1. / len(new_p.values) ** b
                old_post = predicate.get_influence(c) * old_prior
                new_post = new_p.get_influence(c) * new_prior

                if new_post > old_post:
                    predicate = new_p
            return predicate

    def prune(self, predicates, best_point_influence):
        return [p for p in predicates if max(p.point_influence) >= best_point_influence]

    def intersect(self, predicates):
        new_predicates = [p1.merge(p2) for p1, p2 in itertools.combinations(predicates, 2) if not p1.is_adjacent(p2)]
        new_predicates_filtered = ([next(v) for k, v in itertools.groupby(new_predicates, lambda x: x.query)])
        return new_predicates_filtered

    def prune_results(self, results, best_influence, c, SMALL_NUM):
        return [a for b in [r for r in results if r[-1].get_influence(c) >= best_influence - SMALL_NUM] for a in b]

    def set_influence_equal(self, p1, p2):
        x1 = np.log(p1.point_influence.sum())
        x2 = np.log(p2.point_influence.sum())
        s1 = np.log(p1.size)
        s2 = np.log(p2.size)
        return (x1 - x2) / (s1 - s2)

    def strip_cont_predicate(self, predicate, base_predicate):
        for min_val, max_val in base_predicate.values:
            single_predicates = [self.get_base_predicate(base_predicate.feature, v) for v in range(min_val, max_val + 1)]
            while not single_predicates[0].intersects(predicate):
                single_predicates.pop(0)
            while not single_predicates[-1].intersects(predicate):
                single_predicates.pop(-1)
            new_base_predicate = single_predicates[0]
            for i in range(1, len(single_predicates)):
                new_base_predicate = new_base_predicate.merge(single_predicates[i])
            return new_base_predicate

    def strip_disc_predicate(self, predicate, base_predicate):
        single_predicates = [self.get_base_predicate(base_predicate.feature, v) for v in base_predicate.values]
        new_base_p = []
        for p in single_predicates:
            if p.intersects(predicate):
                new_base_p.append(p)
        new_base_predicate = single_predicates[0]
        for i in range(1, len(single_predicates)):
            new_base_predicate = new_base_predicate.merge(single_predicates[i])
        return new_base_predicate

    def strip_predicate(self, predicate):
        if type(predicate) == ContBasePredicate:
            return predicate
        elif type(predicate) == DiscBasePredicate:
            return predicate
        else:
            new_base_predicates = []
            for base_predicate in predicate.base_predicates:
                if type(base_predicate) == ContBasePredicate:
                    new_base_predicates.append(self.strip_cont_predicate(predicate, base_predicate))
                elif type(base_predicate) == DiscBasePredicate:
                    new_base_predicates.append(self.strip_disc_predicate(predicate, base_predicate))
            new_predicate = new_base_predicates[0]
            for i in range(1, len(new_base_predicates)):
                new_predicate = new_predicate.merge(new_base_predicates[i])
            return new_predicate

    def extend_cont_predicate(self, predicate, base_predicate):
        feature_max = max({k:v for k,v in self.predicate_map.items() if k[0] == base_predicate.feature}, key=lambda x: x[1])[1]
        min_val = base_predicate.values[0][0]
        max_val = base_predicate.values[-1][-1]

        new_predicate = copy.deepcopy(predicate)
        if min_val > 0:
            new_left_predicate = self.get_predicate(base_predicate.feature, [(0, min_val-1)])
            merged_left = new_left_predicate.merge(new_predicate)
            if len(merged_left.selected_index) == len(predicate.selected_index):
                new_predicate = merged_left

        if max_val < feature_max:
            new_right_predicate = self.get_predicate(base_predicate.feature, [(max_val+1, feature_max)])
            merged_right = new_right_predicate.merge(new_predicate)
            if len(merged_right.selected_index) == len(predicate.selected_index):
                new_predicate = merged_right
        return new_predicate

    def extend_predicate(self, predicate):
        if type(predicate) != CompoundPredicate:
            return predicate
        else:
            new_predicate = copy.deepcopy(predicate)
            for base_predicate in new_predicate.base_predicates:
                if type(base_predicate) == ContBasePredicate:
                    new_predicate = self.extend_cont_predicate(predicate, base_predicate)
        return new_predicate

    def clean_predicate(self, predicate, c, b):
        predicate = self.merge_predicate(predicate, c=c, b=b)
        predicate = self.strip_predicate(predicate)
        predicate = self.extend_predicate(predicate)
        return predicate

    def get_c_scale(self, predicates, eps=10**-3):
        sorted_predicates = sorted(predicates, key=lambda x: x.get_influence(1), reverse=True)
        next_worst_predicate = sorted_predicates[0]
        for p in sorted_predicates[1:-1]:
            next_worst_predicate = next_worst_predicate.merge(p)
        worst_predicate = next_worst_predicate.merge(sorted_predicates[-1])

        c_min = self.set_influence_equal(worst_predicate, next_worst_predicate) - eps
        return c_min

    def rescale(self, val, in_min, in_max, out_min, out_max):
        return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))

    def search_predicates(self, predicates, c, b, quantile, maxiters=10, best_influence=-np.inf, verbose=False):
        best_predicates = []
        SMALL_NUM = 10**-4
        best_influence -= SMALL_NUM

        for i in range(maxiters):
            filtered_predicates = [p for p in predicates if (p.get_influence(c) > best_influence) or
                                   (np.any([sorted(p.features) == sorted(best_p.features) for best_p in best_predicates])
                                   and p.get_influence(c) >= best_influence-SMALL_NUM)]

            if len(filtered_predicates) == 0:
                found_features = list(set([a for b in [best_p.features for best_p in best_predicates] for a in b]))
                predicates = [p for p in predicates if not p.has_features(found_features)]
                if verbose: print(best_predicates)
                return [self.clean_predicate(p, c=c, b=b) for p in best_predicates], predicates, best_influence

            threshold = np.quantile([p.get_influence(c) for p in filtered_predicates], quantile)
            merged = [p for p in filtered_predicates if p.get_influence(c) >= threshold]
            predicates = self.merge_adjacent(merged, c)

            if verbose:
                print('iter ' + str(i))
                for p in sorted(predicates, key=lambda x: x.get_influence(c), reverse=True):
                    print(p, p.get_influence(c))
                print()

            best_predicate = max(predicates, key=lambda x: x.get_influence(c))
            best_influence = best_predicate.get_influence(c)
            best_point_influence = max(best_predicate.point_influence)
            best_predicates = [p for p in predicates if p.get_influence(c) >= best_influence - SMALL_NUM]

            predicates = self.prune(predicates, best_point_influence)
            predicates = self.intersect(predicates)

        found_features = list(set([a for b in [best_p.features for best_p in best_predicates] for a in b]))
        predicates = [p for p in predicates if not p.has_features(found_features)]
        if verbose: print(best_predicates)
        return [self.clean_predicate(p, c=c, b=b) for p in best_predicates], predicates, best_influence

    def search_features(self, features, index, c, b, quantile, maxiters=10, verbose=False):
        predicates = copy.deepcopy(self.predicates)
        if features is not None:
            predicates = [p for p in predicates if p.feature in features]
        if index is not None:
            predicates = [p for p in predicates if not set(index).isdisjoint(p.selected_index)]

        all_found_predicates = []
        best_influence = -np.inf
        while len(predicates) > 0:
            found_predicates, predicates, best_influence = self.search_predicates(predicates, c, b, quantile,
                                                            maxiters=maxiters, best_influence=best_influence,
                                                            verbose=verbose)
            all_found_predicates += found_predicates

        final_predicates = [self.predicate_data.disc_predicate_to_cont(p) for p in all_found_predicates]
        return final_predicates

    # def search_features(self, features=None, index=None, c=None, b=None, quantile=None, maxiters=10, verbose=False):
    #     if c is None:
    #         c = self.c
    #     if b is None:
    #         b = self.b
    #     if quantile is None:
    #         quantile = self.quantile
    #     if features is None:
    #         features = self.all_features
    #     SMALL_NUM = 10**-4
    #     predicates = [p for p in self.predicates if p.feature in features]
    #
    #     if index is not None:
    #         predicates = [p for p in predicates if not set(index).isdisjoint(p.selected_index)]
    #
    #     c_min = self.get_c_scale(predicates)
    #     scaled_c = self.rescale(c, 0, 1, c_min, 1)
    #
    #     best_influence = -np.inf
    #     best_predicate = None
    #     for i in range(maxiters):
    #         # if i > 0:
    #         #     for p in sorted(predicates, key=lambda x: x.get_influence(scaled_c))[:20]:
    #         #         print(p, p.get_influence(scaled_c), best_predicate, best_influence)
    #         #     print()
    #
    #         predicates = [p for p in predicates if p.get_influence(scaled_c) >= best_influence - SMALL_NUM]
    #         if len(predicates) == 0:
    #             merged_best_predicate = self.merge_predicate(best_predicate, scaled_c, b)
    #             return merged_best_predicate
    #         threshold = np.quantile([p.get_influence(scaled_c) for p in predicates], quantile)
    #         predicates = [p for p in predicates if p.get_influence(scaled_c) >= threshold]
    #         predicates = [self.merge_predicate(p, scaled_c, b) for p in predicates]
    #
    #         # print(f'merging {len(predicates)}')
    #         # if len(predicates) > 200:
    #         #     for p in predicates:
    #         #         print(p)
    #         merged = self.merge_adjacent(predicates, scaled_c)
    #         if verbose:
    #             print(best_influence)
    #             for p in merged:
    #                 print(p, p.get_influence(scaled_c))
    #             print()
    #         # print('done merging')
    #         #
    #         # for p in sorted(merged, key=lambda x: x.get_influence(c), reverse=True):
    #         #     print(p, p.get_influence(scaled_c))
    #         # print()
    #
    #         best_predicate = max(merged, key=lambda x: x.get_influence(scaled_c))
    #         best_influence = best_predicate.get_influence(scaled_c)
    #         best_point_influence = min(best_predicate.point_influence)
    #
    #         pruned = self.prune(merged, best_point_influence)
    #         predicates = self.intersect(pruned)
    #
    #     merged_best_predicate = self.merge_predicate(best_predicate, scaled_c, b)
    #     return merged_best_predicate
    #
    # def search(self, targets=None, index=None, c=None, b=None, quantile=None, maxiters=2, verbose=False):
    #     if targets is None:
    #         targets = self.all_features
    #     target_predicates = self.search_features(targets, index, c, b, quantile, maxiters)
    #     other_features = [f for f in self.all_features if f not in targets]
    #     if len(other_features) == 0:
    #         return [target_predicates]
    #     else:
    #         other_predicates = self.search_features(other_features, index, c, b, quantile, maxiters, verbose)
    #         predicates = [target_predicates, other_predicates]
    #         return predicates