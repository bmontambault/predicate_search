import numpy as np

class Aggregate(object):

    def __call__(self, data):
        return self.aggregate(data)

    def aggregate(self, data):
        raise NotImplemented

class Density(Aggregate):

    def __init__(self, model, targets):
        self.model = model
        self.targets = targets

    def aggregate(self, data):
        logp = self.model.logp(data[self.targets])
        joint_logp = logp.sum()
        return -joint_logp

class Threshold(Aggregate):

    def __init__(self, model, targets, threshold):
        self.model = model
        self.targets = targets
        self.threshold = threshold

    def aggregate(self, data):
        distance = self.model.distance(data[self.targets])
        over_threshold = (distance > self.threshold).sum()
        return over_threshold