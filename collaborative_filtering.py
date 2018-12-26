#!/usr/bin/env python3

"""
Author: Suraj Regmi
Date: 25th December, 2018
Description: Implementation of collaborative filtering from scratch using NumPy
Application Case: Recommending books to the users based on their ratings on other books i.e user-based recommendation
system
"""

# Check if the NumPy module is available.
try:
    import numpy as np
except ImportError:
    print("This implementation requires the NumPy module.\nPlease install NumPy by:\npip(or pip3) install NumPy")
    exit(0)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def evaluator(U, V, ratings):
    """
    Returns root mean square error of the predicted ratings (using U and V) with respect to actual ratings (ratings)
    :param ratings: actual ratings dataframe having user_id, book_id and rating columns
    :return: root mean squared error of the predicted ratings with respect to the actual ratings
    """
    predicted_ratings = []
    for i in ratings.index:

        book_id = ratings.loc[i, 'book_id']
        user_id = ratings.loc[i, 'user_id']

        predicted_rating = np.dot(V[book_id-1, :], U[user_id-1, :])
        predicted_ratings.append(predicted_rating)

    return np.sqrt(mean_squared_error(ratings['rating'], predicted_ratings))


def collaborative_filtering(R, U, V, K, epochs=5, alpha=0.01, beta=0.02):
    """
    Symbols and Definition:
    n_u : No of users
    n_b : No of books

    @INPUT:
        R     : Ratings matrix having dimension (n_u x n_b)
        K     : No of latent features to be used
        U     : User features matrix (n_u x K)
        V     : Book features matrix (n_b x K)
        steps : the maximum number of iterations to perform the optimisation
        alpha : the learning rate
        beta  : the regularization parameter

    @OUTPUT:
        Rating matrix with all the predicted ratings.
    """

    train_ratings, test_ratings = train_test_split(R, test_size=0.02)

    R = train_ratings

    V = V.T

    # Run steps number of iteration training
    for epoch in range(1, epochs+1):

        # Do the random shuffling so as to make SGD equivalent
        R = R.sample(frac=1)

        for l in R.index:
            i = R.loc[l, 'user_id'] - 1
            j = R.loc[l, 'book_id'] - 1

            for k in range(K):

                # Update weights according to gradient descent algorithm
                U[i, k] = U[i, k] - alpha * (
                        2 * (np.dot(U[i, :], V[:, j]) - R.loc[l, 'rating']) * V[k, j] + beta * 2 * U[i, k]
                )
                V[k, j] = V[k, j] - alpha * (2 * (np.dot(U[i, :], V[:, j]) - R.loc[l, 'rating']) * U[i, k] + beta * 2 * V[k, j])
        print("Epoch {} completed.".format(epoch))
        print("RMSE for training set:", evaluator(U, V.T, train_ratings))
        print('RMSE for test set:', evaluator(U, V.T, test_ratings))

    return U, V
