import numpy as np
import pandas as pd

class BasePredicate:

    def __init__(self, feature, values, data, dtype):
        self.size = 1
        if dtype == 'continuous':
            predicate = ContPredicate(feature, values, data)
        elif dtype == 'date':
            predicate = DatePredicate(feature, values, data)
        elif dtype == 'discrete':
            predicate = DiscPredicate(feature, values, data)
        self.__class__ = predicate.__class__
        self.__dict__.update(predicate.__dict__)

    def get_mask(self, query):
        return self.data.eval(query)

    def contains(self, predicate):
        return np.in1d(predicate.indices, self.indices).all()

    def overlaps(self, predicate):
        return np.isin(self.indices, predicate.indices).sum()

    def __repr__(self):
        return self.query

class ContPredicate(BasePredicate):

    def __init__(self, feature, values, data):
        self.size = 1
        self.feature = feature
        self.values = values
        self.data = data
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
        return interval_a[1] >= interval_b[0]

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
        values = self.merge_predicate_intervals(predicate)
        return ContPredicate(self.feature, values, self.data)

class DatePredicate(ContPredicate):

    def __init__(self, feature, values, data):
        super().__init__(feature, values, data)
        self.values = (pd.to_datetime(self.values[0]), pd.to_datetime(self.values[1]))

    def get_query(self):
        return f"{self.feature} > '{str(self.values[0])}' and {self.feature} <= '{str(self.values[1])}'"

    def merge(self, predicate):
        values = self.merge_predicate_intervals(predicate)
        return DatePredicate(self.feature, values, self.data)

class DiscPredicate(BasePredicate):

    def __init__(self, feature, values, data):
        self.size = 1
        self.feature = feature
        self.values = values
        self.data = data
        self.query = self.get_query()
        self.mask = self.get_mask(self.query)
        self.indices = self.data[self.mask].index

    def get_query(self):
        return f"{self.feature} in {self.values}"

    def is_adjacent(self, predicate):
        return self.feature == predicate.feature

    def merge(self, predicate):
        values = sorted(list(set(self.values + predicate.values)))
        return DiscPredicate(self.feature, values, self.data)

class Predicate(BasePredicate):

    def __init__(self, feature_values_list, data, feature_dtype=None):
        self.feature_values_list = feature_values_list
        self.features = list(set([a for b in [list(feature_values.keys()) for feature_values in feature_values_list]
                                  for a in b]))
        self.data = data
        if feature_dtype is None:
            self.feature_dtype = {feature: 'continuous' for feature in self.features}
        else:
            self.feature_dtype = feature_dtype
        predicate = self.get_disjunction()
        self.__class__ = predicate.__class__
        self.__dict__.update(predicate.__dict__)

    def get_conjunction(self, feature_values):
        feature_predicate = {feature: BasePredicate(feature, values, self.data, self.feature_dtype[feature]) for feature, values
                             in feature_values.items()}
        return Conjunction(feature_predicate, self.data)

    def get_disjunction(self):
        conjunctions = [self.get_conjunction(feature_values) for feature_values in self.feature_values_list]
        return Disjunction(conjunctions, self.data)

class Disjunction(BasePredicate):

    def __init__(self, base_predicates, data, mask=None, indices=None, size=None):
        self.base_predicates = base_predicates
        self.features = [base_predicate.features for base_predicate in self.base_predicates]
        self.data = data
        self.query = self.get_query()
        if mask is None:
            if len(base_predicates) > 0:
                self.mask = pd.concat([p.mask for p in base_predicates], axis=1).sum(axis=1).astype(bool)
                self.indices = np.unique(np.concatenate([p.indices for p in self.base_predicates])).astype(int)
            else:
                self.mask = pd.Series(np.zeros(data.shape[0])).astype(bool)
                self.indices = np.array([])
            self.size = sum([predicate.size for predicate in self.base_predicates])
        else:
            self.mask = mask
            self.indices = indices
            self.size = size

    def get_query(self):
        return " or ".join([f"({predicate.query})" for predicate in self.base_predicates])

    def merge(self, predicate):
        base_predicates = self.base_predicates + [predicate]
        mask = (self.mask + predicate.mask).astype(bool)
        indices = np.unique(np.concatenate([self.indices, predicate.indices])).astype(int)
        size = self.size + predicate.size
        return Disjunction(base_predicates, self.data, mask, indices, size)

class Conjunction(BasePredicate):

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
        self.size = len(self.feature_predicate)

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
            feature_predicate[predicate.feature] = feature_predicate[predicate.feature].merge(predicate)
        else:
            feature_mask[predicate.feature] = predicate.mask
            feature_predicate[predicate.feature] = predicate
        return Conjunction(feature_predicate, self.data, feature_mask.astype(bool))

    def merge(self, predicate):
        if type(self) == type(predicate):
            return self.merge_conjunction(predicate)
        else:
            return self.merge_base(predicate)

    def is_adjacent(self, predicate):
        if type(self) == type(predicate):
            for feature in self.features:
                if not self.feature_predicate[feature].is_adjacent(predicate.feature_predicate[feature]):
                    return False
            return True
        else:
            return self.feature_predicate[predicate.feature].is_adjacent(predicate)

    def trim(self):
        feature_predicate = {}
        for feature, predicate in self.feature_predicate.items():
            predicate_class = type(predicate)
            if predicate_class == DiscPredicate:
                values = predicate.values
            else:
                feature_data = self.data.loc[self.mask][feature]
                values = [feature_data.min(), feature_data.max()]
            feature_predicate[feature] = predicate_class(feature, values, self.data)
        return Conjunction(feature_predicate, self.data, self.feature_mask)