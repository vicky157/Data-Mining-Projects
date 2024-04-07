import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score


def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))

def get_neighbors(X_train, test_row, k):
    distances = [(index, euclidean_distance(test_row, train_row)) for index, train_row in enumerate(X_train)]
    distances.sort(key=lambda tup: tup[1])
    neighbors = [index for index, _ in distances[:k]]
    return neighbors

def predict_classification(X_train, y_train, test_row, k):
    neighbors_indices = get_neighbors(X_train, test_row, k)
    output_values = [y_train.iloc[i] for i in neighbors_indices]
    prediction = Counter(output_values).most_common(1)[0][0]
    return prediction

def precision_recall_f1(y_true, y_pred):
    true_pos = np.sum((y_pred == 1) & (y_true == 1))
    false_pos = np.sum((y_pred == 1) & (y_true == 0))
    false_neg = np.sum((y_pred == 0) & (y_true == 1))
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)

data_path = 'set2/project1_dataset2.txt'  
columns = ['Age', 'Tobacco', 'LDL', 'Adiposity', 'FamHist', 'TypeA', 'Obesity', 'Alcohol', 'Age_at_onset', 'CHD']
df = pd.read_csv(data_path, sep="\t", header=None, names=columns)
df = pd.get_dummies(df, columns=['FamHist'])
# df['FamHist'] = LabelEncoder().fit_transform(df['FamHist'])
X = normalize(df.drop('CHD', axis=1))
y = df['CHD']
X = StandardScaler().fit_transform(X)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
k_range = range(1, 100)
best_k = 0
best_accuracy = 0

all_y_true = []
all_y_pred = []

for k in k_range:
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    fold_y_true, fold_y_pred = [], []  
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_pred = [predict_classification(X_train, y_train, row, k) for row in X_test]
        acc = np.mean(y_pred == y_test)
        prec, rec, f1 = precision_recall_f1(y_test, y_pred)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        fold_y_true.extend(y_test)
        fold_y_pred.extend(y_pred)

    all_y_true.extend(fold_y_true)
    all_y_pred.extend(fold_y_pred)

    avg_accuracy = np.mean(accuracies)
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_k = k




print(f"Best k: {best_k} with average accuracy: {best_accuracy:.4f}")
print_confusion_matrix(all_y_true, all_y_pred)
print_classification_report(all_y_true, all_y_pred)

weighted_precision = precision_score(all_y_true, all_y_pred, average='weighted')
weighted_recall = recall_score(all_y_true, all_y_pred, average='weighted')
weighted_f1 = f1_score(all_y_true, all_y_pred, average='weighted')

print(f"Precision: {weighted_precision:.4f}")
print(f"Recall: {weighted_recall:.4f}")
print(f"F1-Score: {weighted_f1:.4f}")