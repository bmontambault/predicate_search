

class Aggregate(object):

    def __call__(self, data):
        return self.aggregate(data)

    def aggregate(self, data):
        raise NotImplemented

class Density(Aggregate):

    def __init__(self, model):
        self.model = model

    def aggregate(self, data):
        logp = self.model.logp(data)
        joint_logp = logp.sum()
        return -joint_logp

class Threshold(Aggregate):

    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def aggregate(self, data):
        distance = self.model.distance(data)
        over_threshold = (distance > self.threshold).sum()
        return over_threshold