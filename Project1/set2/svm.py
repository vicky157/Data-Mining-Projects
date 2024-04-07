import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import norm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import numpy as np
from sklearn.preprocessing import normalize

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
    
data_path = 'set2/project1_dataset2.txt'
columns = ['Age', 'Tobacco', 'LDL', 'Adiposity', 'FamHist', 'TypeA', 'Obesity', 'Alcohol', 'Age_at_onset', 'CHD']
data = pd.read_csv(data_path, sep="\t", header=None, names=columns)
data = pd.get_dummies(data, columns=['FamHist'])

X = data.drop('CHD', axis=1)
y = data['CHD']
scaler = StandardScaler()

X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
y_converted = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_converted, test_size=0.2, random_state=42)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies, precisions, recalls, f1s = [], [], [], []

for train_index, test_index in kf.split(X_normalized):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y_converted[train_index], y_converted[test_index]

    svm = LinearSVM()
    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))
    precisions.append(precision_score(y_test, predictions, average='binary'))
    recalls.append(recall_score(y_test, predictions, average='binary'))
    f1s.append(f1_score(y_test, predictions, average='binary'))

print(f'Average Accuracy: {np.mean(accuracies)}')
print(f'Average Precision: {np.mean(precisions)}')
print(f'Average Recall: {np.mean(recalls)}')
print(f'Average F1-Score: {np.mean(f1s)}')





