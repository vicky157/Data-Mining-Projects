import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, NMF, KNNBasic, accuracy
from surprise.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from time import time

ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
ratings['timestamp'] = MinMaxScaler().fit_transform(ratings[['timestamp']])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

param_grid = {
    'SVD': {
        'n_factors': [50, 100,150],
        'n_epochs': [5,10,20, 30],
        'lr_all': [0.002,0.005, 0.01],
        'reg_all': [0.02, 0.1,0.4]
    },
    'NMF': {
        'n_factors': [15, 20],
        'n_epochs': [50, 70],
        'reg_pu': [0.06, 0.1],
        'reg_qi': [0.06, 0.1]
    },
    'KNNBasic': {
        'k': [20, 40],
        'min_k': [5, 10],
        'sim_options': {'name': ['msd', 'cosine'], 'user_based': [False]}
    }
}

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    precision = sum(precisions.values()) / len(precisions)
    recall = sum(recalls.values()) / len(recalls)
    return precision, recall

results = {}
for algo in param_grid:
    gs = GridSearchCV(eval(algo), param_grid[algo], measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    best_model = gs.best_estimator['rmse']
    
    trainset = data.build_full_trainset()
    best_model.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions = best_model.test(testset)
    precision, recall = precision_recall_at_k(predictions)
    
    cv_results = cross_validate(best_model, data, measures=['rmse', 'mae'], cv=5, return_train_measures=True, n_jobs=-1, verbose=False)
    fit_time = np.mean(cv_results['fit_time'])
    test_time = np.mean(cv_results['test_time'])

    results[algo] = {
        'RMSE': np.mean(cv_results['test_rmse']),
        'MAE': np.mean(cv_results['test_mae']),
        'Precision': precision,
        'Recall': recall,
        'Fit Time': fit_time,
        'Test Time': test_time
    }

for algo, metrics in results.items():
    print(f"{algo} Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
    print("\n")

for algo, metrics in results.items():
    print(f"{algo} Results:")
    for key, value in metrics.items():
        std_dev = np.std(cv_results[f'test_{key.lower()}']) if key in ['RMSE', 'MAE'] else 0  # No std dev for precision, recall, times
        print(f"{key}: {value:.3f} ± {std_dev:.3f}")
    print("\n")