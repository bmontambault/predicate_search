import pandas as pd

from .models import RobustCov, IsoForest, LOF

class Score:

    models = {'robust_cov': RobustCov, 'iso_forest': IsoForest, 'lof': LOF}

    def __init__(self, model, data, targets, **kwargs):
        self.model = self.models[model](**kwargs)
        self.data = data
        self.targets = targets
        self.fit()
        self.score = self.get_score()

    def fit(self):
        self.model.fit(self.data[self.targets])

    def get_score(self):
        return pd.Series(self.model.score(self.data[self.targets]), index=self.data.index)

    def likelihood_influence(self, indices, specificity):
        return self.score.loc[indices].sum() / (len(indices)**specificity)