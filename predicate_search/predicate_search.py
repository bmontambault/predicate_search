import itertools
import numpy as np
import copy

from .predicate import Predicate

class PredicateSearch:

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
            if col in disc_cols:
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
            if k in self.disc_cols:
                min_val, max_val = self.col_min_max[k]
                cont_v_lower = list(np.array(v) * (max_val - min_val) / (self.bins - 1) + min_val)
                cont_v_upper = list((np.array(v)+1) * (max_val - min_val) / (self.bins - 1) + min_val)
                cont_v = list(zip(cont_v_lower, cont_v_upper))
                cont_v_merged = self.merge_cont_values(cont_v)
                feature_values_dict[k] = cont_v_merged
        return Predicate(feature_values_dict)

    def get_predicates(self):
        return [a for b in [[Predicate({col: [val]}) for val in range(self.bins)] for col in self.disc_data.columns] for a in b]

    def get_influence(self, predicate, c):
        filtered_data = predicate.remove(self.data)
        size = len(self.data) - len(filtered_data)
        agg_delta = self.aggregate(self.data) - self.aggregate(filtered_data)
        influence = agg_delta / size ** c
        return influence

    def get_top_n(self, predicates, n, c):
        sorted_predicates = sorted(predicates, key=lambda p: self.get_influence(p, c), reverse=True)
        top_predicates = sorted_predicates[:n]
        return top_predicates

    def merge_adjacent(self, predicate, other_predicates, c):
        SMALL_NUM = 10**-3
        influence = self.get_influence(predicate, c)
        adjacent = [p for p in other_predicates if p.is_adjacent(predicate)]
        for i in range(len(adjacent)):
            if other_predicates[i].is_adjacent(predicate):
                new_predicate = predicate.merge(other_predicates[i])
                new_influence = self.get_influence(new_predicate, c)
                if new_influence >= influence - SMALL_NUM:
                    del other_predicates[i]
                    return self.merge_adjacent(new_predicate, other_predicates, c)
        return predicate

    def merge_all_adjacent(self, predicates, c):
        all_merged = []
        sorted_predicates = sorted(predicates, key=lambda p: self.get_influence(p, c), reverse=True)
        i = 0
        while i < len(sorted_predicates):
            predicate = sorted_predicates[i]
            other_predicates = sorted_predicates[i+1:]
            merged = self.merge_adjacent(predicate, other_predicates, c)
            all_merged.append(merged)
            i+=1
        return all_merged

    def intersect(self, predicates, c):
        all_intersected = [(p1, p2, p1.merge(p2)) for p1, p2 in itertools.combinations(predicates, 2)]
        intersected = [p[2] for p in all_intersected if self.get_influence(p[2], c) > max(self.get_influence(p[0], c), self.get_influence(p[1], c))]
        return intersected

    def search_predicates(self, predicates, c, topn=5, maxiters=10):
        best_inf = -np.inf
        top_predicates = []
        for i in range(maxiters):
            merged_predicates = self.merge_all_adjacent(predicates, c)
            new_top_predicates = self.get_top_n(merged_predicates, topn, c)
            new_best_inf = self.get_influence(new_top_predicates[-1], c)
            if new_best_inf <= best_inf:
                return top_predicates
            else:
                best_inf = new_best_inf
                top_predicates = new_top_predicates
            intersected_predicates = self.intersect(merged_predicates, c)
            predicates = intersected_predicates + top_predicates