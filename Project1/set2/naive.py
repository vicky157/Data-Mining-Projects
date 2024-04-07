import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

data_path = 'set2/project1_dataset2.txt'
columns = ['Age', 'Tobacco', 'LDL', 'Adiposity', 'FamHist', 'TypeA', 'Obesity', 'Alcohol', 'Age_at_onset', 'CHD']
data = pd.read_csv(data_path, sep="\t", header=None, names=columns)
data = pd.get_dummies(data, columns=['FamHist'])

X = data.drop('CHD', axis=1)
y = data['CHD']
scaler = StandardScaler()

X_normalized = scaler.fit_transform(X)

threshold=0.5
y = [1 if value > threshold else 0 for value in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0),
                'prior': X_c.shape[0] / X.shape[0]
            }
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            y_pred = [self._predict(row[1].to_numpy().astype(float)) for row in X.iterrows()]
        else:
            raise TypeError("X must be a pandas DataFrame")
        return np.array(y_pred)

    
    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.parameters[c]['prior'])
            class_conditional = np.sum(np.log(self._pdf(c, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, class_, x):
        mean = self.parameters[class_]['mean']
        var = self.parameters[class_]['var']
        numerator = np.exp(-(x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


nb = NaiveBayes()

def cross_validate_naive_bayes(X, y, cv=10):
    y = np.array(y)  
    kf = KFold(n_splits=cv)
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        nb.fit(X_train_fold, y_train_fold)
        y_pred_fold = nb.predict(X_test_fold)

        accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
        precisions.append(precision_score(y_test_fold, y_pred_fold, average='binary'))
        recalls.append(recall_score(y_test_fold, y_pred_fold, average='binary'))
        f1_scores.append(f1_score(y_test_fold, y_pred_fold, average='binary'))
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1_score = np.mean(f1_scores)

    return avg_accuracy, avg_precision, avg_recall, avg_f1_score

avg_accuracy, avg_precision, avg_recall, avg_f1_score = cross_validate_naive_bayes(X, y)

print(f"Average Accuracy: {avg_accuracy}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1-Score: {avg_f1_score}")
