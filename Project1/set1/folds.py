import numpy as np

def KFold(X, y, n_splits=10, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled, y_shuffled = X[indices], y[indices]
    fold_sizes = np.full(n_splits, X.shape[0] // n_splits, dtype=int)
    fold_sizes[:X.shape[0] % n_splits] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append((indices[start:stop], np.concatenate([indices[:start], indices[stop:]])))
        current = stop
    return folds