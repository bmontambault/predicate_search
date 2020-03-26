from .model import RobustNormal
from .predicate_data import PredicateData
from .predicate_search import PredicateSearch
from .predicate import BasePredicate, ContBasePredicate, DiscBasePredicate

class AnomalyDetection:

    def __init__(self, c=.5):
        self.c = c

    def fit(self, data):
        self.data = data
        model = RobustNormal()
        model.fit(data)
        self.mean = model.params['mean']
        self.cov = model.params['cov']
        logp = model.score(data)

        self.predicate_data = PredicateData(data)
        predicates = self.predicate_data.get_base_predicates(logp)
        self.predicate_search = PredicateSearch(predicates)

    def search(self, targets=None, index=None, c=None, maxiters=10):
        if targets is None:
            targets = self.data.columns.tolist()
        if c is None:
            c = self.c

        raw_predicate = self.predicate_search.search(c=c, targets=targets, index=index, maxiters=maxiters)
        predicate = [self.predicate_data.disc_predicate_to_cont(p) for p in raw_predicate]
        return predicate
