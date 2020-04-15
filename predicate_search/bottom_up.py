import itertools
import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
import copy

from .predicate import Predicate

class BottomUp:

    def __init__(self, data, aggregate, disc_cols=[], bins=100):
        self.data = data
        self.disc_data, self.col_min_max = self.cont_to_disc(self.data, disc_cols, bins)

        self.aggregate = aggregate
        self.disc_cols = disc_cols
        self.bins = bins

    def cont_to_disc(self, data, disc_cols, bins):
        disc_data = data.copy()
        col_min_max = {}
        for col in disc_data.columns:
            if col not in disc_cols:
                min_val = disc_data[col].min()
                max_val = disc_data[col].max()
                disc_data[col] = ((self.data[col] - min_val) / (max_val - min_val) * (bins - 1)).astype(int)
                col_min_max[col] = (min_val, max_val)
        return disc_data, col_min_max

    def merge_overlapping_values(self, values_a, values_b):
        if values_a[1] > values_b[1]:
            return [values_a]
        elif values_a[1] >= values_b[0]:
            return [(values_a[0], values_b[1])]
        else:
            return [values_a, values_b]

    def merge_cont_values(self, values):
        len_values = len(values)
        i = 0
        while i < len_values - 1:
            values_a = values[i]
            values_b = values[i + 1]
            merged = self.merge_overlapping_values(values_a, values_b)
            if len(merged) == 1:
                len_values -= 1
                values[i + 1] = merged[0]
                del values[i]
            else:
                i += 1
        return values

    def disc_to_cont(self, predicate):
        feature_values_dict = predicate.feature_values_dict.copy()
        for k,v in feature_values_dict.items():
            if k not in self.disc_cols:
                min_val, max_val = self.col_min_max[k]
                cont_v_lower = list(np.array(v) * (max_val - min_val) / (self.bins - 1) + min_val)
                cont_v_upper = list(np.clip((np.array(v) + 1) * (max_val - min_val) / (self.bins - 1) + min_val, a_min=None, a_max=max_val))
                cont_v = list(zip(cont_v_lower, cont_v_upper))
                cont_v_merged = self.merge_cont_values(cont_v)
                feature_values_dict[k] = cont_v_merged
        return Predicate(feature_values_dict)

    def get_predicates(self):
        return [a for b in [[Predicate({col: [val]}, self.disc_cols) for val in range(self.bins)] for col in self.disc_data.columns] for a in b]

    def get_influence(self, predicate, c):
        filtered_data = self.data.iloc[predicate.remove_index(self.disc_data)]
        size = len(self.data) - len(filtered_data)
        agg_delta = self.aggregate(self.data) - self.aggregate(filtered_data)
        if size > 0:
            influence = agg_delta / size ** c
        else:
            influence = 0
        return influence

    def get_best_single_tuple_influence(self, predicate):
        filtered_data = self.data.iloc[predicate.remove_index(self.disc_data)]
        best_index, best_score = self.aggregate.single_tuple_aggregate(filtered_data)
        agg_delta = self.aggregate(self.data) - best_score
        return best_index, agg_delta

    def get_top_n(self, predicates, n, c):
        sorted_predicates = sorted(predicates, key=lambda p: self.get_influence(p, c), reverse=True)
        filtered_predicates = [sorted_predicates[i] for i in range(len(sorted_predicates)) if not np.any([sorted_predicates[i].isin(p) for p in sorted_predicates[:i]])]
        top_predicates = filtered_predicates[:n]
        return top_predicates

    def prune(self, predicates, data, index):
        pruned = [p for p in predicates if p.contains_index(data, index)]
        return pruned

    def prune_top_n(self, predicates, top_predicates, data):
        best_tuple_index = [self.get_best_single_tuple_influence(p)[0] for p in top_predicates]

        print('pruning top')
        for p in predicates:
            print(p, p.contains_index(data, best_tuple_index))
        print()

        pruned = [p for p in predicates if p.contains_index(data, best_tuple_index)]
        return pruned

    def merge_adjacent(self, predicate, other_predicates, c):
        SMALL_NUM = 10**-3
        influence = self.get_influence(predicate, c)
        for i in range(len(other_predicates)):
            if other_predicates[i].is_adjacent(predicate):
                new_predicate = predicate.merge(other_predicates[i])
                new_influence = self.get_influence(new_predicate, c)
                print(f'{predicate}:', influence, f'{new_predicate}:', new_influence)
                if new_influence >= influence - SMALL_NUM:
                    del other_predicates[i]
                    return self.merge_adjacent(new_predicate, other_predicates, c)
        return predicate, other_predicates

    def merge_all_adjacent(self, predicates, c):
        print('merging')
        all_merged = []
        sorted_predicates = sorted(predicates, key=lambda p: self.get_influence(p, c), reverse=True)
        i = 0
        while i < len(sorted_predicates):
            predicate = sorted_predicates[0]
            other_predicates = sorted_predicates[1:]
            merged, sorted_predicates = self.merge_adjacent(predicate, other_predicates, c)
            all_merged.append(merged)
            i+=1
        all_merged += sorted_predicates
        return all_merged

    def intersect(self, predicates, c):
        all_intersected = [(p1, p2, p1.merge(p2)) for p1, p2 in itertools.combinations(predicates, 2)]
        intersected = [p[2] for p in all_intersected if self.get_influence(p[2], c) > max(self.get_influence(p[0], c), self.get_influence(p[1], c))]
        return intersected

    def find_top_predicates(self, predicates, c=.8, index=None, topn=5, max_merged=100, maxiters=10):
        top_predicates = []
        for i in range(maxiters):
            print('iter: '+ str(i))
            if index is None:
                pruned_predicates = predicates
            else:
                pruned_predicates = self.prune(predicates, self.disc_data, index)
            if len(pruned_predicates) == 0:
                return top_predicates

            merged_predicates = self.merge_all_adjacent([p for p in pruned_predicates if self.get_influence(p, c) > 0], c)
            print()
            print('merged:')
            for p in sorted(merged_predicates, key=lambda x: self.get_influence(x, c), reverse=True):
                print(p, self.get_influence(p, c))

            new_top_predicates = self.get_top_n(merged_predicates + top_predicates, topn, c)
            if new_top_predicates == top_predicates:
                return top_predicates
            else:
                top_predicates = new_top_predicates

            top_merged_predicates = sorted(merged_predicates, key=lambda x: self.get_influence(x, c), reverse=True)[:max_merged]

            print('intersecting', len(top_merged_predicates))
            intersected_predicates = self.intersect(top_merged_predicates, c)
            print('done intersecting')

            print()
            print('intersected:')
            for p in sorted(intersected_predicates, key=lambda x: self.get_influence(x, c), reverse=True):
                print(p, self.get_influence(p, c))
            print()

            predicates = intersected_predicates
        return top_predicates

    def group_ranges(self, vals):
        ranges = []
        for k, g in groupby(enumerate(vals), lambda x: x[0] - x[1]):
            group = (map(itemgetter(1), g))
            group = list(map(int, group))
            ranges.append((group[0], group[-1]))
        return ranges

    def merge_intervals(self, predicate, c):
        features_values = predicate.feature_values_dict.items()
        influence = self.get_influence(predicate, c)
        for k, v in features_values:
            if k not in predicate.disc_cols:
                missing = [i for i in range(self.bins) if i not in v]
                missing_ranges = self.group_ranges(missing)
                for interval in missing_ranges:
                    values = list(range(interval[0], interval[1] + 1))
                    new_predicate = Predicate({k: values}, disc_cols=predicate.disc_cols)
                    merged_predicate = predicate.merge(new_predicate)
                    new_influence = self.get_influence(merged_predicate, c)
                    if new_influence >= influence:
                        influence = new_influence
                        predicate = merged_predicate
        return predicate

    def find_predicates(self, predicates, c=.8, index=None, topn=5, max_merged=100, maxiters=10):
        top_predicates = self.find_top_predicates(predicates, c, index, topn, max_merged, maxiters)
        merged_predicates = [self.merge_intervals(p, c) for p in top_predicates]
        continuous_predicates = [self.disc_to_cont(p) for p in merged_predicates]
        return continuous_predicates