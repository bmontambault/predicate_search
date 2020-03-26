import numpy as np
import itertools
import pandas as pd

from .predicate import BasePredicate, ContBasePredicate, DiscBasePredicate, CompoundPredicate

class PredicateData:

    def __init__(self, data, disc_cols=[], bins=100):
        self.data = data
        self.features = list(self.data.columns)
        self.disc_cols = disc_cols
        self.bins = bins
        self.min_val = data.min()
        self.max_val = data.max()
        self.disc_data = pd.DataFrame()
        for col in data.columns:
            if col not in disc_cols:
                self.disc_data[col] = self.cont_to_disc(col)
            else:
                self.disc_data[col] = self.data[col]
        self.disc_range = {col: pd.Series({d: (self.data[self.disc_data[col] == d][col].min(), self.data[self.disc_data[col] == d][col].max())
                                     for d in self.disc_data[col].unique()}) for col in self.disc_data.columns}
        self.adj_matrix = {col: self.get_adj_matrix(col) for col in self.data.columns}

    def cont_to_disc(self, col, d=None):
        if d is None:
            d = self.data[col]
        disc_d = ((d - self.min_val[col]) / (self.max_val[col] - self.min_val[col]) * (self.bins - 1)).astype(int)
        return disc_d

    def disc_to_cont(self, col, d=None):
        if d is None:
            d = self.disc_data[col]
        return self.disc_range[col][d]

    def disc_base_predicate_to_cont(self, predicate):
        if type(predicate) == DiscBasePredicate:
            return predicate
        else:
            feature = predicate.feature
            values = [(self.disc_to_cont(feature, a)[0], self.disc_to_cont(feature, b)[1]) for a,b in predicate.values]
            return ContBasePredicate(feature, values, predicate.selected_index, predicate.logp, predicate.adj_matrix)

    def disc_predicate_to_cont(self, predicate):
        if predicate.__class__.__base__ == BasePredicate:
            return self.disc_base_predicate_to_cont(predicate)
        else:
            base_predicates = [self.disc_base_predicate_to_cont(p) for p in predicate.base_predicates]
        return CompoundPredicate(base_predicates)

    def get_adj_matrix(self, col):
        unique = sorted(self.disc_data[col].unique())
        val_range = np.arange(len(unique))
        adj = np.abs(val_range[:,None] - val_range[None,:]) <= 1
        adj_matrix = pd.DataFrame(adj, index=unique, columns=unique)
        return adj_matrix

    def get_base_predicates(self, logp=None, index=None):
        if logp is None:
            logp = np.zeros(self.data.shape[0])
        if index is not None:
            logp = logp[np.array(index)]
            disc_data = self.disc_data.iloc[index].reset_index(drop=True)
        else:
            disc_data = self.disc_data

        predicates = []
        for col in disc_data.columns:
            values = sorted(disc_data[col].unique())
            for val in values:
                selected_index = np.array(disc_data[disc_data[col] == val].index)
                if col in self.disc_cols:
                    predicate = DiscBasePredicate(col, [val], selected_index, logp)
                else:
                    predicate = ContBasePredicate(col, [(val, val)], selected_index, logp, self.adj_matrix[col])
                predicates.append(predicate)
        return predicates