import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import SVD
import numpy as np
DATA_PATH = "./ml-latest-small/all_del.csv"
data = pd.read_csv(DATA_PATH)
# Basic data analysis
data_info = data.info()
data_describe = data.describe()
missing_values = data.isnull().sum()

data_info, data_describe, missing_values



# Prepare the data for the recommendation system
reader = Reader(rating_scale=(0, 5))
print(reader)
data_surprise = Dataset.load_from_df(data[['userId', 'movieId', 'adjusted_rating']], reader)

# Split the dataset into training and test sets (75% training, 25% testing)
trainset, testset = train_test_split(data_surprise, test_size=0.25)

# Use a matrix factorization model, such as SVD (Singular Value Decomposition)
model = SVD()

# Train the model on the training set
model.fit(trainset.build_full_trainset())

# Predict ratings for the test set
predictions = model.test(testset)

# Compute RMSE (Root Mean Squared Error)
rmse = mean_squared_error([pred.r_ui for pred in predictions], [pred.est for pred in predictions], squared=False)

rmse


# Creating a sparse matrix for userId, movieId, and adjusted_rating
# Note: We need to adjust the indices to be zero-based
user_ids = data['userId'].unique()
movie_ids = data['movieId'].unique()
user_id_map = {id: index for index, id in enumerate(user_ids)}
movie_id_map = {id: index for index, id in enumerate(movie_ids)}

# Map the userId and movieId to zero-based indices for matrix operations
data['userId'] = data['userId'].map(user_id_map)
data['movieId'] = data['movieId'].map(movie_id_map)

# Create a sparse matrix
sparse_matrix = csr_matrix((data['adjusted_rating'], (data['userId'], data['movieId'])), shape=(len(user_ids), len(movie_ids)))

# Apply SVD (Singular Value Decomposition)
n_components = 20  # Number of singular values and vectors to compute
svd = TruncatedSVD(n_components=n_components)
matrix_reduced = svd.fit_transform(sparse_matrix)

# Explained variance ratio of the SVD
explained_variance = svd.explained_variance_ratio_.sum()

print(matrix_reduced.shape, explained_variance)
# For a user and a movie that are not in the training set, we need to use a different approach.
# One common approach is to compute the predicted rating based on similar users and/or similar movies.
# User and movie IDs for prediction
user_id = 1
movie_id = 2

# Check if the user_id and movie_id are in the mapping (adjust for zero-based index)
user_idx = user_id_map.get(user_id - 1)
movie_idx = movie_id_map.get(movie_id - 1)

if user_idx is not None and movie_idx is not None:
    # Predict the rating using the SVD reduced matrix
    # We use the dot product of the user's and movie's features
    predicted_rating = np.dot(matrix_reduced[user_idx, :], svd.components_[:, movie_idx])
else:
    predicted_rating = None

predicted_rating

