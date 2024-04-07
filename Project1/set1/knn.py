import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn(X_train, y_train, X_test, k):
    y_pred = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        y_pred.append(most_common[0][0])
    return np.array(y_pred)

def precision_recall_f1(y_true, y_pred):
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    actual_positives = np.sum(y_true == 1)
    
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

file_path = 'set1/project1_dataset1.txt'  
df = pd.read_csv(file_path, sep='\t', header=None)
df = df.sample(frac=1).reset_index(drop=True)


numerical_cols = df.columns[:-1]  
df_normalized = df.copy()
for col in numerical_cols:
    df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()


X = df_normalized.iloc[:, :-1].values
y = df_normalized.iloc[:, -1].values
n_splits = 10
fold_size = len(df) // n_splits

k_range = range(1, 26)  
best_k = 0
best_accuracy = 0

for k in k_range:
    fold_accuracies = []
    for fold in range(n_splits):
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_test, y_test = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        y_pred = knn(X_train, y_train, X_test, k)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
    
    avg_accuracy = np.mean(fold_accuracies)
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_k = k

print(f"Optimal k: {best_k} with average accuracy: {best_accuracy:.4f}")


precisions, recalls, f1_scores = [], [], []
for fold in range(n_splits):
    start, end = fold * fold_size, (fold + 1) * fold_size
    X_test, y_test = X[start:end], y[start:end]
    X_train = np.concatenate((X[:start], X[end:]), axis=0)
    y_train = np.concatenate((y[:start], y[end:]), axis=0)

    y_pred = knn(X_train, y_train, X_test, best_k)
    prec, rec, f1 = precision_recall_f1(y_test, y_pred)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)


average_precision = np.mean(precisions)
average_recall = np.mean(recalls)
average_f1_score = np.mean(f1_scores)

print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average F1 Score: {average_f1_score:.4f}")
