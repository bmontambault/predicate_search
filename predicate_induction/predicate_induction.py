import pandas as pd
import numpy as np
import copy

from .predicate import ContPredicate, DiscPredicate, DatePredicate, Conjunction, Disjunction
from .score import Score

class PredicateInduction:

    def __init__(self, data, model, targets, specificity, bins=100, **kwargs):
        self.data = data
        self.score = Score(model, data, targets, **kwargs)
        self.targets = targets
        self.bins = bins
        self.dtypes = {col: self.infer_dtype(self.data[col]) for col in self.data.columns}
        self.specificity = specificity
        self.set_predicates()

    def drop_indices(self, indices):
        self.data = self.data[~self.data.index.isin(indices)]
        self.score.score = self.score.score[~self.score.score.index.isin(indices)]
        self.set_predicates()

    def set_predicates(self):
        self.predicate_scores = {}
        self.base_predicates = self.get_base_predicates()
        self.predicates = [a for b in [[Conjunction({predicates[i].feature: predicates[i]}, self.data)
                            for i in range(len(predicates))] for predicates in self.base_predicates.values()] for a in b]
        self.predicates = self.intersect_predicates(self.predicates)
        self.merged_predicates = {k: self.merge_same_features(v, self.specificity) if len(k) > 1 else v for k, v in self.predicates.items()}
        self.predicates = [a for b in [v for v in self.merged_predicates.values()] for a in b]
        self.predicates = sorted(self.predicates, key=lambda x: self.get_predicate_score(x), reverse=True)

    def add_feature(self, predicate, features=None):
        best_score = self.get_predicate_score(predicate)
        best_predicate = None
        for feature, base_predicates in self.base_predicates.items():
            if feature not in predicate.features and (features is None or feature in features):
                for base_predicate in base_predicates:
                    new_predicate = predicate.merge(base_predicate)
                    new_score = self.get_predicate_score(new_predicate)
                    if new_score > best_score:
                        best_score = new_score
                        best_predicate = new_predicate
        if best_predicate is None:
            return predicate
        else:
            return best_predicate

    def expand_predicates(self, predicate, predicates):
        if len(predicates) == 0:
            return predicate
        for i in range(len(predicates)):
            if predicate.is_adjacent(predicates[i]):
                new_predicate = predicate.merge(predicates[i])
                if self.get_predicate_score(new_predicate) >= self.get_predicate_score(predicate) -10**-10:
                    new_predicates = predicates[:i] + predicates[i+1:]
                    return self.expand_predicates(new_predicate, new_predicates)
        return predicate

    def expand_feature(self, predicate, feature):
        base_predicates = self.base_predicates[feature]
        return self.expand_predicates(predicate, base_predicates)

    def expand_features(self, predicate):
        for feature in predicate.features:
            predicate = self.expand_feature(predicate, feature)
        return predicate

    def maximize_predicate(self, predicate, features=None):
        # print('original', predicate, self.get_predicate_score(predicate))
        new_predicate = self.add_feature(predicate, features)
        # print('new', new_predicate, self.get_predicate_score(new_predicate))
        if self.get_predicate_score(new_predicate) > self.get_predicate_score(predicate):
            predicate = self.expand_features(new_predicate)
            return self.maximize_predicate(predicate, features)
        else:
            return predicate

    def infer_dtype(self, d, origin=1970):
        if pd.to_datetime(d).min().year > origin:
            return 'date'
        else:
            if pd.to_numeric(d, errors='coerce').notnull().all():
                if np.array_equal(d, d.astype(int)):
                    return 'discrete'
                else:
                    return 'continuous'
            else:
                return 'discrete'

    def get_base_predicates(self):
        features_predicates = {}
        for feature, dtype in self.dtypes.items():
            if dtype == 'discrete':
                predicates = [DiscPredicate(feature, [val], self.data) for val in self.data[feature].unique().tolist()]
            elif dtype in ['continuous', 'date']:
                values = pd.cut(self.data[feature], bins=self.bins, right=True)
                intervals = pd.DataFrame([(interval.left, interval.right) for interval in values.sort_values().unique()])
                adjacent = intervals.rename(columns={0:'left', 1:'right'})
                adjacent['left'] = adjacent['left'].shift(-1)
                intervals = [list(a) for a in list(intervals.dropna().to_records(index=False))]
                adjacent = [list(a) for a in list(adjacent.dropna().to_records(index=False))]
                if dtype == 'continuous':
                    predicates = [ContPredicate(feature, interval, adjacent, self.data, values) for interval in intervals]
                elif dtype == 'date':
                    predicates = [DatePredicate(feature, interval, adjacent, self.data, values) for interval in intervals]
            sorted_predicates = sorted(predicates, key=lambda p: self.get_predicate_score(p), reverse=True)
            features_predicates[feature] = self.merge_same_features(sorted_predicates, self.specificity)
        return features_predicates

    def set_predicate_score(self, predicate):
        self.predicate_scores[predicate.query] = self.score.likelihood_influence(predicate.indices, self.specificity)

    def get_predicate_score(self, predicate):
        if predicate in self.predicate_scores.keys():
            return self.predicate_scores[predicate.query]
        else:
            self.set_predicate_score(predicate)
            return self.predicate_scores[predicate.query]

    def merge_same_features(self, predicates, specificity):
        predicates = predicates.copy()
        merged_predicates = []
        while len(predicates) > 0:
            p = predicates.pop(0)
            p, predicates = self.merge_predicate_list(p, predicates, specificity)
            merged_predicates.append(p)
        return merged_predicates

    def merge_predicate_list(self, p, predicates, specificity):
        for i in range(len(predicates)):
            new_p = predicates[i]
            # print(p, new_p, p.is_adjacent(new_p))
            if p.is_adjacent(new_p):
                merged_p = p.merge(new_p)
                # print(self.get_predicate_score(p), self.get_predicate_score(merged_p))
                if self.get_predicate_score(merged_p) >= self.get_predicate_score(p) -10**-10:
                    del predicates[i]
                    return self.merge_predicate_list(merged_p, predicates, specificity)
        return p, predicates

    def merge_predicate(self, predicate):
        new_base_predicates = []
        for base_predicate in predicate:
            feature_predicates = self.base_predicates[base_predicate.feature]
            for feature_predicate in feature_predicates:
                new_predicate = predicate.merge(feature_predicate)
                if self.get_predicate_score(new_predicate) >= self.get_predicate_score(predicate):
                    predicate = new_predicate
            new_base_predicates.append(predicate)
        return Conjunction(new_base_predicates, self.data)

    def insert_predicate(self, predicate_list, predicate):
        for i in range(len(predicate_list)):
            if self.get_predicate_score(predicate) > self.get_predicate_score(predicate_list[i]):
                predicate_list.insert(i, predicate)
                return None
        predicate_list.append(predicate)

    def intersect_predicates(self, predicates):
        new_predicates = {}
        for predicate in predicates:
            if predicate.features in new_predicates:
                new_predicates[predicate.features].append(predicate)
            else:
                new_predicates[predicate.features] = [predicate]

        for i in range(len(predicates)):
            for j in range(i+1, len(predicates)):
                if predicates[i].feature_predicate.keys() != predicates[j].feature_predicate.keys():
                    new_predicate = predicates[i].merge(predicates[j])
                    new_score = self.get_predicate_score(new_predicate)
                    old_score_a = self.get_predicate_score(predicates[i])
                    old_score_b = self.get_predicate_score(predicates[j])
                    if new_score -10**-10 > old_score_a and new_score -10**-10 > old_score_b:
                        if new_predicate.features in new_predicates.keys():
                            self.insert_predicate(new_predicates[new_predicate.features], new_predicate)
                        else:
                            new_predicates[new_predicate.features] = [new_predicate]
                        if predicates[i] in new_predicates[predicates[i].features]:
                            new_predicates[predicates[i].features].remove(predicates[i])
                        if predicates[j] in new_predicates[predicates[j].features]:
                            new_predicates[predicates[j].features].remove(predicates[j])
        return new_predicates

    def search(self):
        predicate = self.predicates[0]
        for p in self.predicates[1:]:
            new_predicate = predicate.join(p)
            new_score = self.get_predicate_score(new_predicate)
            if new_score >= self.get_predicate_score(predicate) -10**-10:
                predicate = new_predicate
            else:
                return predicate
        return predicate

    def search_all(self, maxiters=100):
        full_data = self.data.copy()
        full_score = self.score.score.copy()
        all_predicates = []
        old_score = np.inf
        s = self.score.score.var()
        for i in range(maxiters):
            score = self.score.score.sum() / (len(self.score.score))
            if score < old_score - s:
                old_score = score
                predicate = self.search()
                self.drop_indices(predicate.indices)
                all_predicates.append(predicate)
            else:
                break
        self.data = full_data
        self.score.score = full_score
        joined_predicate = all_predicates[0]
        for p in all_predicates[1:]:
            joined_predicate = joined_predicate.join(p)
        return joined_predicate

    def get_data(self, predicate):
        if len(self.targets) == 1:
            y_feature = self.targets[0]
        else:
            y_feature = None
        feature_d = {}
        for feature, p in predicate.feature_predicate.items():
            mask = predicate.feature_mask[[col for col in predicate.feature_mask.columns if col != feature]].prod(axis=1).astype(bool)
            mask = mask.reindex(self.data.index).fillna(True)
            if y_feature is None or y_feature == feature:
                d = pd.Series(self.score.score.loc[mask].values, index=self.data.loc[mask][feature])
            else:
                d = pd.Series(self.data.loc[mask][y_feature].values, index=self.data.loc[mask][feature])
            df = d.sort_index().reset_index().rename(columns={0: [y_feature, 'score'][feature == y_feature]})
            df['anomaly'] = df.eval(predicate.feature_predicate[feature].query).astype(int)
            feature_d[feature] = df
        return feature_d