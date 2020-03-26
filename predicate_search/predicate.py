import numpy as np

class Predicate(object):

    # def get_avg_logp_change(self, logp):
    #     if len(self.selected_index) == 0:
    #         return 0
    #     point_logp = logp[self.selected_index]
    #     size = len(point_logp)
    #     avg_logp = (logp.sum() - point_logp.sum()) / (len(logp) - size)
    #     avg_logp_change = logp.mean() - avg_logp
    #     return avg_logp_change
    #
    # def get_point_influence(self, logp):
    #     if len(self.selected_index) == 0:
    #         return 0
    #     point_logp = logp[self.selected_index]
    #     point_influence = logp.mean() - ((logp.sum() - point_logp) / (len(logp) - 1))
    #     return point_influence
    #
    # def get_influence(self, c):
    #     if self.size == 0:
    #         return 0
    #     return self.avg_logp_change / self.size**c

    def get_point_influence(self, logp):
        if len(self.selected_index) == 0:
            return 0
        point_influence = -logp[self.selected_index]
        return point_influence

    def get_influence(self, c):
        if self.size == 0:
            return 0
        return self.point_influence.sum() / self.size**c

    def intersects(self, predicate):
        return len(set(list(predicate.selected_index)).intersection(set(list(self.selected_index)))) > 0

    def has_features(self, features):
        return len(set(list(self.features)).intersection(set(list(features)))) > 0


class BasePredicate(Predicate):
    
    def __init__(self, feature, values, selected_index, logp):
        self.feature = feature
        self.features = [feature]
        self.values = values
        self.selected_index = selected_index
        self.size = len(selected_index)
        self.logp = logp
        self.query = self.get_query()
        #self.avg_logp_change = self.get_avg_logp_change(logp)
        self.point_influence = self.get_point_influence(logp)

    def merge(self, predicate):
        if self.__class__.__base__ == BasePredicate and predicate.__class__.__base__ == BasePredicate:
            if self.feature == predicate.feature:
                selected_index = np.array(list(set(list(self.selected_index) + list(predicate.selected_index))))
                if type(self) == ContBasePredicate:
                    merged_interval_lists = self.merge_interval_lists(self.values.copy(), predicate.values.copy())
                    merged_intervals = self.merge_intervals(merged_interval_lists)
                    return ContBasePredicate(self.feature, merged_intervals, selected_index, self.logp,
                                             self.adj_matrix)
                elif type(self) == DiscBasePredicate:
                    return DiscBasePredicate(self.feature, self.values + predicate.values, selected_index, self.logp)
            else:
                return CompoundPredicate([self, predicate])
        elif type(self) == CompoundPredicate:
            return self.merge(predicate)
        elif type(predicate) == CompoundPredicate:
            return predicate.merge(self)

    def get_obj(self):
        return {self.feature: self.values}

    def __repr__(self):
        return f"{self.feature}: {self.values}"
        
class ContBasePredicate(BasePredicate):
    
    def __init__(self, feature, values, selected_index, logp, adj_matrix):
        self.adj_matrix = adj_matrix
        super().__init__(feature, values, selected_index, logp)

    def merge_interval_lists(self, intervals_a, intervals_b):
        intervals = []
        while len(intervals_a) > 0 and len(intervals_b) > 0:
            if intervals_a[0][0] < intervals_b[0][0]:
                intervals.append(intervals_a.pop(0))
            else:
                intervals.append(intervals_b.pop(0))
        if len(intervals_a) > 0:
            intervals += intervals_a
        if len(intervals_b) > 0:
            intervals += intervals_b
        return intervals

    def merge_overlapping_intervals(self, interval_a, interval_b):
        # first interval contains second interval: return first interval
        if interval_a[1] > interval_b[1]:
            return [interval_a]
        # first interval overlaps second interval: return start of first interval and end of second interval
        # endpoints are adjacent: return start of first interval and end of second interval
        elif interval_a[1] > interval_b[0] or interval_a[1]+1 == interval_b[0] or (interval_a[1] in self.adj_matrix.keys()
                                                                                    and interval_b[0] in self.adj_matrix[interval_a[1]].keys()
                                                                                    and self.adj_matrix[interval_a[1]][interval_b[0]]):
            return [(interval_a[0], interval_b[1])]
        # intervals not overlapping or adjacent
        else:
            return [interval_a, interval_b]

    def merge_intervals(self, intervals):
        len_intervals = len(intervals)
        i = 0
        while i < len_intervals - 1:
            intervals_a = intervals[i]
            intervals_b = intervals[i + 1]
            merged = self.merge_overlapping_intervals(intervals_a, intervals_b)
            if len(merged) == 1:
                len_intervals -= 1
                intervals[i + 1] = merged[0]
                del intervals[i]
            else:
                i += 1
        return intervals

    def is_adjacent(self, predicate):
        if type(predicate) != ContBasePredicate:
            return False
        if self.feature != predicate.feature:
            return False
        intervals = self.merge_interval_lists(self.values.copy(), predicate.values.copy())
        for i in range(len(intervals)-1):
            if self.adj_matrix[intervals[i][1]].loc[intervals[i+1][0]]:
                return True
        return False

    def get_query(self):
        return " or ".join([f"({self.feature} >= {interval[0]} and {self.feature} <= {interval[1]})"
                            for interval in self.values])

class DiscBasePredicate(BasePredicate):

    def __init__(self, feature, values, selected_index, logp):
        values = list(set(values))
        super().__init__(feature, values, selected_index, logp)

    def is_adjacent(self, predicate):
        return type(predicate) == DiscBasePredicate and self.feature == predicate.feature

    def get_query(self):
        return f"{self.feature} in {self.values}"

class CompoundPredicate(Predicate):

    def __init__(self, base_predicates):
        self.base_predicates = base_predicates
        self.features = [p.feature for p in base_predicates]
        self.values = [a for b in [p.values for p in base_predicates] for a in b]
        self.selected_index = np.array(list(set.intersection(*map(set, [p.selected_index for p in self.base_predicates]))))
        self.size = len(self.selected_index)
        self.logp = self.base_predicates[0].logp
        #self.avg_logp_change = self.get_avg_logp_change(self.logp)
        self.point_influence = self.get_point_influence(self.logp)
        self.query = self.get_query()

    def merge_compound(self, predicate):
        base_predicates = self.base_predicates + predicate.base_predicates
        merged_predicate = base_predicates[0]
        for p in base_predicates[1:]:
            merged_predicate = merged_predicate.merge(p)
        return merged_predicate

    def merge_base(self, predicate):
        if predicate.feature in self.features:
            index = self.features.index(predicate.feature)
            base_predicates = self.base_predicates.copy()
            base_predicates[index] = base_predicates[index].merge(predicate)
        else:
            base_predicates = self.base_predicates + [predicate]
        return CompoundPredicate(base_predicates)

    def merge(self, predicate):
        if type(predicate) == CompoundPredicate:
            return self.merge_compound(predicate)
        else:
            return self.merge_base(predicate)

    def is_adjacent(self, predicate):
        if sorted(self.features) != sorted(predicate.features):
            return False
        sorted_predicates1 = sorted(self.base_predicates, key=lambda p: p.feature)
        sorted_predicates2 = sorted(predicate.base_predicates, key=lambda p: p.feature)
        equal_vals_count = 0
        adjacent_vals_count = 0
        for i in range(len(sorted_predicates1)):
            if sorted_predicates1[i].is_adjacent(sorted_predicates2[i]):
                if sorted_predicates1[i].values == sorted_predicates2[i].values:
                    equal_vals_count += 0
                else:
                    adjacent_vals_count += 0
        #return equal_vals_count > 0 and adjacent_vals_count > 0
        return adjacent_vals_count == 1 and equal_vals_count == len(self.features) - 1

    def get_obj(self):
        return {p.feature: p.values for p in self.base_predicates}

    def get_query(self):
        return " and ".join([f"({p.query})" for p in self.base_predicates])

    def __repr__(self):
        return '[' + ", ".join([f"{p.feature}: {p.values}" for p in sorted(self.base_predicates, key=lambda x: x.feature)]) + ']'