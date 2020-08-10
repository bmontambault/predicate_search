import numpy as np
import pandas as pd

class Predicate(object):

    def get_mask(self, query):
        return self.data.eval(query)

    def join(self, predicate):
        if type(self) == Disjunction and type(predicate) == Disjunction:
            return Disjunction(self.base_predicates + predicate.base_predicates, self.data)
        elif type(self) == Disjunction and type(predicate) != Disjunction:
            return Disjunction(self.base_predicates + [predicate], self.data)
        elif type(predicate) == Disjunction and type(self) != Disjunction:
            return Disjunction(predicate.base_predicates + [self], self.data)
        else:
            return Disjunction([self, predicate], self.data)

    def __repr__(self):
        return self.query

class Disjunction(Predicate):

    def __init__(self, base_predicates, data):
        self.base_predicates = base_predicates
        self.data = data
        self.query = self.get_query()
        self.indices = np.unique(np.concatenate([p.indices for p in self.base_predicates]))

    def get_query(self):
        return " or ".join([f"{predicate.query}" for predicate in self.base_predicates])

class Conjunction(Predicate):

    def __init__(self, feature_predicate, data, feature_mask=None):
        self.data = data
        self.feature_predicate = feature_predicate
        self.features = tuple(sorted(list(self.feature_predicate.keys())))
        if feature_mask is None:
            self.feature_mask = self.get_feature_mask()
        else:
            self.feature_mask = feature_mask

        self.query = self.get_query()
        self.mask = self.feature_mask.prod(axis=1).astype(bool)
        self.indices = self.data[self.mask].index

    def get_query(self):
        return " and ".join([f"{predicate.query}" for predicate in self.feature_predicate.values()])

    def get_feature_mask(self):
        feature_mask = pd.DataFrame()
        for feature, predicate in self.feature_predicate.items():
            feature_mask[feature] = predicate.mask
        return feature_mask

    def merge_conjunction(self, predicate):
        feature_predicate = self.feature_predicate.copy()
        feature_mask = self.feature_mask.astype(int)
        for feature, base_predicate in predicate.feature_predicate.items():
            if feature in feature_predicate:
                feature_mask[feature] += base_predicate.mask.astype(int)
                feature_predicate[feature] = feature_predicate[feature].merge(base_predicate)
            else:
                feature_mask[feature] = base_predicate.mask
                feature_predicate[feature] = base_predicate
        return Conjunction(feature_predicate, self.data, feature_mask.astype(bool))

    def merge_base(self, predicate):
        feature_predicate = self.feature_predicate.copy()
        feature_mask = self.feature_mask.astype(int)
        if predicate.feature in self.feature_mask.columns:
            feature_mask[predicate.feature] += predicate.mask
            feature_predicate[predicate.feature].merge(predicate)
        else:
            feature_mask[predicate.feature] = predicate.mask
            feature_predicate[predicate.feature] = predicate
        return Conjunction(feature_predicate, self.data, feature_mask.astype(bool))

    def merge(self, predicate):
        if type(self) == type(predicate):
            return self.merge_conjunction(predicate)
        else:
            return self.merge_base(predicate)

    def feature_is_adjacent(self, predicate, feature):
        other_features = [f for f in self.features if f != feature]
        for other_feature in other_features:
            if self.feature_predicate[other_feature].values != predicate.feature_predicate[other_feature].values:
                return False
        if type(self.feature_predicate[feature]) == DiscPredicate:
            return True
        else:
            mask = self.feature_mask[other_features].prod(axis=1).astype(bool)
            intervals = pd.DataFrame([(interval.left, interval.right) for interval in self.feature_predicate[feature].binned_data[mask].sort_values().unique()])
            adjacent = intervals.rename(columns={0: 'left', 1: 'right'})
            adjacent['left'] = adjacent['left'].shift(-1)
            adjacent = [list(a) for a in list(adjacent.dropna().to_records(index=False))]
            interval_a = self.feature_predicate[feature].values
            interval_b = predicate.feature_predicate[feature].values
            return [interval_a[0], interval_b[1]] in adjacent or [interval_b[0], interval_a[1]] in adjacent

    def is_adjacent(self, predicate):
        if type(self) == type(predicate):
            for feature in self.features:
                if self.feature_is_adjacent(predicate, feature):
                    return True
            return False
        else:
            return self.feature_predicate[predicate.feature].is_adjacent(predicate)

class ContPredicate(Predicate):

    def __init__(self, feature, values, adjacent, data, binned_data):
        self.feature = feature
        self.values = values
        self.adjacent = adjacent
        self.data = data
        self.binned_data = binned_data
        self.query = self.get_query()

        self.mask = self.get_mask(self.query)
        self.indices = self.data[self.mask].index

    def get_query(self):
        return f"({self.feature} > {self.values[0]} and {self.feature} <= {self.values[1]})"

    def merge_intervals(self, interval_a, interval_b):
        if interval_a[1] > interval_b[1]:
            return interval_a
        else:
            return [interval_a[0], interval_b[1]]

    def is_adjacent(self, predicate):
        if self.values[0] < predicate.values[1]:
            interval_a = self.values
            interval_b = predicate.values
        else:
            interval_b = self.values
            interval_a = predicate.values
        return interval_a[1] >= interval_b[0] or [interval_b[0], interval_a[1]] in self.adjacent

    def merge_predicate_intervals(self, predicate):
        if self.values[0] < predicate.values[1]:
            interval_a = self.values
            interval_b = predicate.values
        else:
            interval_b = self.values
            interval_a = predicate.values
        interval = self.merge_intervals(interval_a, interval_b)
        return interval

    def merge(self, predicate):
        interval = self.merge_predicate_intervals(predicate)
        return ContPredicate(self.feature, interval, self.adjacent, self.data, self.binned_data)

class DatePredicate(ContPredicate):

    def __init__(self, feature, interval, adjacent, data, binned_data):
        super().__init__(feature, interval, adjacent, data, binned_data)
        self.values[0] = pd.to_datetime(self.values[0])
        self.values[1] = pd.to_datetime(self.values[1])
        for adj in self.adjacent:
            adj[0] = pd.to_datetime(adj[0])
            adj[1] = pd.to_datetime(adj[1])

    def get_query(self):
        return f"{self.feature} > '{str(self.values[0])}' and {self.feature} <= '{str(self.values[1])}'"

    def merge(self, predicate):
        interval = self.merge_predicate_intervals(predicate)
        return DatePredicate(self.feature, interval, self.adjacent, self.data, self.binned_data)

class DiscPredicate(Predicate):

    def __init__(self, feature, values, data):
        self.feature = feature
        self.values = values
        self.data = data
        self.query = self.get_query()
        # print(self.query)
        self.mask = self.get_mask(self.query)
        self.indices = self.data[self.mask].index

    def get_query(self):
        return f"{self.feature} in {self.values}"

    def is_adjacent(self, predicate):
        return True

    def merge(self, predicate):
        values = sorted(list(set(self.values + predicate.values)))
        return DiscPredicate(self.feature, values, self.data)