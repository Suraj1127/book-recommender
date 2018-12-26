"""
Author: Suraj Regmi
Date: 26th December, 2018
Description: Generates features of both users as well as books and saves in pickle files.

Pickle files:
Books: features/books.pickle
Users: features/users.pickle
"""
from collaborative_filtering import collaborative_filtering

import pickle

import numpy as np
import pandas as pd


def save_features(u, v):
    """
    Save book features and user features in pickle files.
    :param u: users' features
    :param v: books' features
    """
    with open('features/users.pickle', 'wb') as f:
        pickle.dump(u, f)

    with open('features/books.pickle', 'wb') as f:
        pickle.dump(v, f)


if __name__ == "__main__":

    ratings = pd.read_csv('data/raw/ratings.csv')

    n_u = ratings['user_id'].nunique()
    n_b = ratings['book_id'].nunique()
    K = 10

    U = np.random.rand(n_u, K)
    V = np.random.rand(n_b, K)

    # Take 2 latent features and train
    U, V = collaborative_filtering(ratings, U, V, K)

    # Take transpose of V for uniformity of shapes
    V = V.T

    # Save the features
    save_features(U, V)