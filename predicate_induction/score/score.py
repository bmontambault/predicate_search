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
        score = pd.Series(self.model.score(self.data[self.targets]), index=self.data.index)
        normalized_score = score / (score.max() - score.min())
        return normalized_score

    def likelihood_influence(self, indices, specificity):
        return self.score.loc[indices].sum() / (len(indices)**specificity)