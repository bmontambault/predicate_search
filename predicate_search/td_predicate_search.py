import itertools
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree

from .predicate import DiscBasePredicate, ContBasePredicate

class TDPredicateSearch:

    def __init__(self, data, logp):
        self.data = data
        self.logp = logp
        self.influence = logp.mean() - ((logp.sum() - logp) / (len(logp) - 1))

    def search_features(self, features=None, max_predicates=5):
        if features is None:
            features = self.data.columns
        data = self.data[features]
        clf = DecisionTreeRegressor(max_leaf_nodes=max_predicates*2)
        clf.fit(data.values, self.logp.values)

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)



# class Node:
#
#     def __init__(self, data, influence, query_str=''):
#         self.data = data
#         self.influence = influence
#         self.query_str = query_str
#
#     def get_var_reduction(self, feature, value):
#         d = self.data[feature]
#         initial_variance = self.influence.var()
#         left_node = self.influence[d[d <= value].index]
#         right_node = self.influence[d[d > value].index]
#         var_reduction = (left_node.var() + right_node.var())
#         return var_reduction
#
#     def get_split_scores(self, feature):
#         d = self.data[feature]
#         splits = sorted(d.unique())[1:-1]
#         var_reduc = [self.get_var_reduction(feature, value) for value in splits]
#         return splits, var_reduc
#
#     def get_best_value(self, feature):
#         splits, var_reduc = self.get_split_scores(feature)
#         if len(splits) == 0:
#             return None, None, 0
#         split_value_index = np.argmax(var_reduc)
#         split_value = splits[split_value_index]
#         split_score = var_reduc[split_value_index]
#         return feature, split_value, split_score
#
#     def get_best_feature(self):
#         feature_values_scores = [self.get_best_value(feature) for feature in self.data.columns]
#         feature, value, score = max(feature_values_scores, key=lambda x: x[2])
#         return feature, value, score
#
#     def split(self, feature, value):
#         left_data = self.data[self.data[feature] <= value]
#         right_data = self.data[self.data[feature] > value]
#
#         if self.query_str != '':
#             query_str = self.query_str + ' and '
#         else:
#             query_str = self.query_str
#         left_query_str = query_str + f"{feature} <= {value}"
#         right_query_str = query_str + f"{feature} > {value}"
#
#         return (Node(left_data.reset_index(drop=True), self.influence[left_data.index], left_query_str),
#                 Node(right_data.reset_index(drop=True), self.influence[right_data.index], right_query_str))

# class TDPredicateSearch:
#
#     def __init__(self, data, influence):
#         self.data = data
#         self.influence = influence
#
#     def search(self, maxiters=1, threshold=0):
#         leaves = [Node(self.data, self.influence)]
#         dead_leaves = []
#         for i in range(maxiters):
#             j = 0
#             while j < len(leaves):
#                 node = leaves[j]
#                 feature, value, score = node.get_best_feature()
#                 if score > threshold:
#                     leaves[j] = node.split(feature, value)
#                     j += 1
#                 else:
#                     dead_leaves.append(leaves.pop(j))
#             leaves = [a for b in leaves for a in b]
#         return leaves, dead_leaves