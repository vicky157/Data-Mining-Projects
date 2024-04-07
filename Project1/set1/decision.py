from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

file_path = 'set1/project1_dataset1.txt'
df = pd.read_csv(file_path, sep='\t', header=None)  

numerical_cols = df.columns

df_min_max_scaled = df.copy()
for col in numerical_cols:
    min_col = df_min_max_scaled[col].min()
    max_col = df_min_max_scaled[col].max()
    df_min_max_scaled[col] = (df_min_max_scaled[col] - min_col) / (max_col - min_col)

df_z_score_normalized = df.copy()
for col in numerical_cols:
    mean_col = df_z_score_normalized[col].mean()
    std_col = df_z_score_normalized[col].std()
    df_z_score_normalized[col] = (df_z_score_normalized[col] - mean_col) / std_col

X = df_z_score_normalized.iloc[:, :-1].values
y = df_z_score_normalized.iloc[:, -1].values

threshold=0.5

y = [1 if value > threshold else 0 for value in y]

dataset = np.column_stack((X,y)).tolist()
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies, precisions, recalls, f1s = [], [], [], []

for train_index, test_index in kf.split(dataset):
    train, test = [dataset[i] for i in train_index], [dataset[i] for i in test_index]
    tree = build_tree(train, max_depth=5, min_size=10) 
    y_true, y_pred = [], []
    for row in test:
        prediction = predict(tree, row)
        y_true.append(row[-1])
        y_pred.append(prediction)

    accuracies.append(accuracy_score(y_true, y_pred))
    precisions.append(precision_score(y_true, y_pred, average='macro', zero_division=0))
    recalls.append(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1s.append(f1_score(y_true, y_pred, average='macro', zero_division=0))

avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1 = np.mean(f1s)

print(f"Average Accuracy: {avg_accuracy}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1-Score: {avg_f1}")
