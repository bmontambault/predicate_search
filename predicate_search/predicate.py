import numpy as np

class Predicate(object):

    def __init__(self, feature_values_dict, disc_cols=[]):
        self.feature_values_dict = feature_values_dict
        self.query = self.get_query()
        self.disc_cols = disc_cols

    def get_feature_query(self, feature, values):
        if type(values[0]) == tuple:
            return "(" + " or ".join([f"({feature} >= {val[0]} and {feature} <= {val[1]})" for val in values]) + ")"
        else:
            return f"{feature} in {values}"

    def get_query(self):
        return " and ".join([self.get_feature_query(k, v) for k,v in self.feature_values_dict.items()])

    def filter_index(self, data):
        return data.query(self.query).index

    def contains_index(self, data, index):
        return len(set(list(self.filter_index(data))).intersection(set(index))) > 0

    def isin(self, predicate):
        for k, v in self.feature_values_dict.items():
            if k not in predicate.feature_values_dict.keys() or len([val for val in v if val not in predicate.feature_values_dict[k]]) > 0:
                return False
        return True

    def remove_index(self, data):
        return data.query(f"~({self.query})").index

    def size(self, data):
        return len(self.filter(data))

    def values_adjacent(self, values1, values2):
        return np.min(np.abs(np.array(values1)[:,None] - np.array(values2)[None,:])) <= 1

    def is_adjacent(self, predicate):
        if sorted(self.feature_values_dict.keys()) != sorted(predicate.feature_values_dict.keys()):
            return False
        for feature in self.feature_values_dict.keys():
            if feature in self.disc_cols:
                return True
            elif self.values_adjacent(self.feature_values_dict[feature], predicate.feature_values_dict[feature]):
                return True
        return False

    def merge(self, predicate):
        feature_values_dict = self.feature_values_dict.copy()
        for k,v in predicate.feature_values_dict.items():
            if k in feature_values_dict:
                feature_values_dict[k] = sorted(list(set(feature_values_dict[k] + predicate.feature_values_dict[k])))
            else:
                feature_values_dict[k] = predicate.feature_values_dict[k]
        return Predicate(feature_values_dict, self.disc_cols)

    def __repr__(self):
        return self.query