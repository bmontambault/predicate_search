import numpy as np

class Predicate(object):

    def __init__(self, feature_values_dict, disc_cols=[]):
        self.feature_values_dict = feature_values_dict
        self.query = self.get_query()
        self.disc_cols = disc_cols

    def get_query(self):
        " and ".join([f"{k} in {v}" for k,v in self.feature_values_dict.items()])

    def filter(self, data):
        return data.query(self.query)

    def remove(self, data):
        return data.query(f"~({self.query})")

    def size(self, data):
        return len(self.filter(data))

    def values_adjacent(self, values1, values2):
        return np.min(np.abs(np.array(values1)[:,None] - np.array(values2)[None,:])) <= 1

    def is_adjacent(self, predicate):
        common_features = list(set(list(self.feature_values_dict.keys()) + list(predicate.feature_values_dict.keys())))
        for feature in common_features:
            if feature in self.disc_cols:
                return True
            elif self.values_adjacent(self.feature_values_dict[feature], predicate.feature_values_dict[feature]):
                return True
        return False

    def merge(self, predicate):
        feature_values_dict = self.feature_values_dict.copy()
        for k,v in predicate.feature_values_dict:
            if k in feature_values_dict:
                feature_values_dict[k] = list(set(feature_values_dict[k] + predicate.feature_values_dict[k]))
            else:
                feature_values_dict[k] = predicate.feature_values_dict[k]
        return Predicate(feature_values_dict, self.disc_cols)

    def __repr__(self):
        return self.query