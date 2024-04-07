from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def gini_index(groups, classes, weights, dataset):
    n_instances = sum([weights[i] for group in groups for i in group])
    gini = 0.0
    for group in groups:
        group_size = float(len(group))
        if group_size == 0:
            continue
        score = 0.0
        group_weights = sum([weights[i] for i in group])
        for class_val in classes:
            p = sum([weights[i] for i in group if dataset[i][-1] == class_val]) / group_weights
            score += p * p
        gini += (1.0 - score) * (group_weights / n_instances)
    return gini


def test_split(index, value, dataset):
    left, right = list(), list()
    for i in range(len(dataset)):
        if dataset[i][index] < value:
            left.append(i)
        else:
            right.append(i)
    return left, right

def get_split(dataset, weights):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values, weights, dataset)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth, dataset):  
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal([dataset[i] for i in left + right])
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal([dataset[i] for i in left]), to_terminal([dataset[i] for i in right])
        return
    if len(left) <= min_size:
        node['left'] = to_terminal([dataset[i] for i in left])
    else:
        node['left'] = get_split([dataset[i] for i in left], dataset)  
        split(node['left'], max_depth, min_size, depth+1, dataset)  
    if len(right) <= min_size:
        node['right'] = to_terminal([dataset[i] for i in right])
    else:
        node['right'] = get_split([dataset[i] for i in right], dataset)  
        split(node['right'], max_depth, min_size, depth+1, dataset)  

def build_tree(train, max_depth, min_size, sample_weight):
    root = get_split(train, sample_weight)  
    split(root, max_depth, min_size, 1, train)  
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return 1 if node['left'] == 1 else -1
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return 1 if node['right'] == 1 else -1

def ada_boost_train(X_train, y_train, n_trees):
    weights = np.full(len(X_train), 1 / len(X_train))
    trees = []
    tree_weights = []

    dataset = [list(x) + [y] for x, y in zip(X_train, y_train)]

    for _ in range(n_trees):
        tree = build_tree(dataset, 1, 1, weights)
        predictions = np.array([predict(tree, x) for x in X_train])
        misclassified = np.array([int(x) for x in (predictions != y_train)])
        error = np.dot(weights, misclassified) / sum(weights)
        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
        weights *= np.exp(-alpha * y_train * predictions)
        weights /= sum(weights)
        trees.append(tree)
        tree_weights.append(alpha)

    return trees, tree_weights

def ada_boost_predict(X, trees, tree_weights):
    final_predictions = np.zeros(len(X))
    for tree, weight in zip(trees, tree_weights):
        predictions = np.array([predict(tree, x) for x in X])
        final_predictions += weight * predictions
    return np.sign(final_predictions)

data = pd.read_csv('set1/project1_dataset1.txt', sep='\t', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = np.where(y == 0, -1, 1)  
scaler = StandardScaler()
X= scaler.fit_transform(X)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

accuracies, precisions, recalls, f1s = [], [], [], []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    n_trees = 50 
    trees, tree_weights = ada_boost_train(X_train, y_train, n_trees)
    y_pred = ada_boost_predict(X_test, trees, tree_weights)
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))


avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1 = np.mean(f1s)

print(f"Average Accuracy: {avg_accuracy}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1-Score: {avg_f1}")


