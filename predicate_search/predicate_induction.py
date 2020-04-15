from .bottom_up import BottomUp
from .model import RobustNormal
from .aggregate import Density
from .predicate import Predicate

class PredicateInduction:

    def __init__(self, data, disc_cols, model_class=RobustNormal, aggregate=Density):
        self.data = data
        self.disc_cols = disc_cols
        self.model_class = model_class
        self.aggregate = aggregate

        self.min_val, self.max_val, self.norm_data = self.normalize_data(data)
        self.model = self.fit(self.norm_data)

    def normalize_data(self, data):
        min_val = data.min()
        max_val = data.max()
        norm_data = (data.drop(self.disc_cols, axis=1) - min_val) / (max_val - min_val)
        for col in self.disc_cols:
            norm_data[col] = data[col]
        return min_val, max_val, norm_data

    def fit(self, data, **kwargs):
        model = self.model_class()
        model.fit(data.drop(self.disc_cols, axis=1), **kwargs)
        return model

    def rescale_predicate(self, predicate):
        feature_values_dict = predicate.feature_values_dict.copy()
        for k, v in feature_values_dict.items():
            if k not in self.disc_cols:
                min_val = self.min_val[k]
                max_val = self.max_val[k]
                rescaled = [(s * (max_val - min_val) + min_val, e * (max_val - min_val) + min_val) for s, e in v]
                try:
                    rescaled_formatted = [(float(s), float(e)) for s, e in rescaled]
                except:
                    rescaled_formatted = [(f"'{str(s)}'", f"'{str(e)}'") for s, e in rescaled]
                feature_values_dict[k] = rescaled_formatted
        rescaled_predicate = Predicate(feature_values_dict)
        return rescaled_predicate

    def find_predicates(self, targets, index=None, c=.8, topn=5, max_merged=100, maxiters=10):
        aggregate = Density(self.model, targets)
        bottom_up = BottomUp(self.norm_data, aggregate, self.disc_cols)
        predicates = bottom_up.get_predicates()
        found_predicates = bottom_up.find_predicates(predicates, c, index, topn, max_merged, maxiters)
        rescaled_predicates = [self.rescale_predicate(p) for p in found_predicates]
        return rescaled_predicates