import numpy as np

class Aggregate(object):

    def __call__(self, data):
        return self.aggregate(data)

    def aggregate(self, data):
        raise NotImplemented

    def single_tuple_aggregate(self, data):
        score = self.score(data)
        best_index = np.argmax(score)
        best_score = score[best_index]
        return best_index, best_score

class Density(Aggregate):

    def __init__(self, model, targets):
        self.model = model
        self.targets = targets

    def score(self, data):
        logp = -self.model.logp(data[self.targets])
        return logp

    def aggregate(self, data):
        logp = self.score(data)
        joint_logp = logp.sum()
        return joint_logp

class Threshold(Aggregate):

    def __init__(self, model, targets, threshold):
        self.model = model
        self.targets = targets
        self.threshold = threshold

    def score(self, data):
        distance = self.model.distance(data[self.targets])
        return distance

    def aggregate(self, data):
        distance = self.score(data)
        over_threshold = (distance > self.threshold).sum()
        return over_threshold