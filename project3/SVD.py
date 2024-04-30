# import pandas as pd
# from surprise import Dataset, Reader
# from surprise.model_selection import train_test_split
# from surprise import SVD, accuracy

# from surprise.model_selection import cross_validate

# # Load user ratings
# ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# ratings['rating'] = (ratings['rating'] - ratings['rating'].min()) / (ratings['rating'].max() - ratings['rating'].min())

# reader = Reader(rating_scale=(0, 1))
# data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# trainset, testset = train_test_split(data, test_size=0.25)


# model = SVD()

# # Train the model
# model.fit(trainset)

# from surprise.model_selection import GridSearchCV

# param_grid = {
#     'n_factors': [50, 100, 150],
#     'n_epochs': [20, 30], 
#     'lr_all': [0.005, 0.01],
#     'reg_all': [0.02, 0.1]
# }

# gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
# gs.fit(data)

# # Best RMSE score
# print('Best RMSE:', gs.best_score['rmse'])

# # Best model parameters
# print('Best parameters:', gs.best_params['rmse'])
# # Use the best model from grid search
# best_model = gs.best_estimator['rmse']
# best_model.fit(data.build_full_trainset())

# # Predictions on the test set
# predictions = best_model.test(testset)

# # Calculate and print RMSE
# accuracy_rmse = accuracy.rmse(predictions)
# print(f"Test Set RMSE: {accuracy_rmse:.3f}")


# # Using cross-validation to evaluate the model
# results = cross_validate(best_model, data, measures=['RMSE'], cv=5, verbose=True)

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV, cross_validate

# Load user ratings
ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Preprocessing: normalization (scaling ratings between 0 and 1)
ratings['rating'] = (ratings['rating'] - ratings['rating'].min()) / (ratings['rating'].max() - ratings['rating'].min())

# Load the data into Surprise
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# Parameter grid for GridSearchCV
param_grid = {
    'n_factors': [50, 100, 150],
    'n_epochs': [20, 30], 
    'lr_all': [0.005, 0.01],
    'reg_all': [0.02, 0.1]
}

# Initialize and run grid search
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

# Output best RMSE score and parameters
print('Best RMSE:', gs.best_score['rmse'])
print('Best parameters:', gs.best_params['rmse'])

# Use the best model from grid search
best_model = gs.best_estimator['rmse']

# Perform cross-validation with the optimized model
results = cross_validate(best_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

# Output cross-validation results
mean_rmse = np.mean(results['test_rmse'])
std_rmse = np.std(results['test_rmse'])

# Calculate mean and standard deviation for MAE
mean_mae = np.mean(results['test_mae'])
std_mae = np.std(results['test_mae'])

# Calculate mean and standard deviation for fit time
mean_fit_time = np.mean(results['fit_time'])
std_fit_time = np.std(results['fit_time'])

# Calculate mean and standard deviation for test time
mean_test_time = np.mean(results['test_time'])
std_test_time = np.std(results['test_time'])

# Print results
print(f"Mean RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")
print(f"Mean MAE: {mean_mae:.3f} ± {std_mae:.3f}")
print(f"Mean fit time: {mean_fit_time:.3f} seconds ± {std_fit_time:.3f}")
print(f"Mean test time: {mean_test_time:.3f} seconds ± {std_test_time:.3f}")


