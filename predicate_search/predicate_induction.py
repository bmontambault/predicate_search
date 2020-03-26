import numpy as np

from .model import RobustNormal
from .predicate_data import PredicateData
from .predicate_search import PredicateSearch

class PredicateInduction:

    def __init__(self, model=RobustNormal, c=1, b=.1, quantile=0):
        self.model = model
        self.c = c
        self.b = b
        self.quantile = quantile

    def fit(self, data, disc_cols=[], **kwargs):
        self.data = data
        self.features = data.columns
        self.disc_cols = disc_cols
        data_min = data.drop(disc_cols, axis=1).min()
        data_max = data.drop(disc_cols, axis=1).max()

        norm_data = ((data.drop(disc_cols, axis=1) - data_min) / (data_max - data_min)).reset_index(drop=True)
        self.norm_data = norm_data
        for col in disc_cols:
            norm_data[col] = data[col]
        self.m = self.model()
        self.m.fit(norm_data.drop(disc_cols, axis=1), **kwargs)

    def predicate_induction(self, targets=None, threshold=None, c=None, b=None, quantile=None, maxiters=10, verbose=False):
        if c is None:
            c = self.c
        if b is None:
            b = self.b
        if quantile is None:
            quantile = self.quantile

        distances = self.m.get_distances(self.norm_data, targets)
        all_p = []
        data = self.data.copy()
        for i in range(1):
            if threshold is None:
                index = None
            else:
                index = list(data[distances >= threshold].index)
            logp = self.m.score(data, targets)
            predicate_search = PredicateSearch(data, logp, self.disc_cols, c=c, b=b, quantile=quantile)

            # features = [f for f in data.columns if f not in targets]
            p = predicate_search.search_features(features=features, index=index, c=c, b=b, quantile=quantile,
                                                 maxiters=maxiters, verbose=verbose)
            # p = predicate_search.search_features(features=self.features, index=index, c=c, b=b, quantile=quantile,
            #                                      maxiters=maxiters, verbose=verbose)
            index = [a for b in [pi.selected_index for pi in p] for a in b]
            data = data[data.index.isin(index)].reset_index(drop=True)
            all_p += p
        return all_p
