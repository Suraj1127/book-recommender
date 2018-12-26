#!/usr/bin/env python3

"""
Author: Suraj Regmi
Date: 26th December, 2018
Description: Trains the feature vector of the user based on his ratings and recommends the books to the user.
"""

import json
import pickle

import numpy as np


def load_features():
    """
    Load all the features of users and books.
    :return: (U, V), U representing features matrix for users and V representing features matrix for books.
    """
    with open('features/users.pickle', 'rb') as f:
        U = pickle.load(f)

    with open('features/books.pickle', 'rb') as f:
        V = pickle.load(f)

    return U, V


def load_json(filepath):
    with open(filepath) as fp:
        return json.load(fp)


def train_user_features(dictionary, V, alpha=0.01, epochs=100):
    """
    Trains the feature vector for the user based on the ratings given by him.
    :param dictionary: dictionary having ids of books as keys and ratings of books as values
    :param V: Books feature vector
    :param alpha: learning rate
    :param epochs: no of epochs/train_iterations
    :return: feature fector for the user
    """
    user_feature_vector = np.random.randn(1, 10)
    for epoch in range(epochs):
        for book_id in dictionary.keys():
            user_feature_vector -= alpha * 2 * (
                        np.dot(user_feature_vector, V[int(book_id) - 1, :])
                        - dictionary[book_id]
            ) * (V[int(book_id) - 1, :].reshape(1, -1))
    return user_feature_vector


def get_recommendations(ufv, V, no_of_recommendations):
    """
    Returns title of recommended books
    :param ufv: feature vector for the user
    :param V: feature matrix for all the books
    :param no_of_recommendations: no of recommendations to be given
    :return: Titles of recommended books
    """
    predicted_ratings = np.matmul(V, ufv.T).reshape(-1)

    id_recommendations = np.argsort(predicted_ratings)[-no_of_recommendations:][::-1]
    books_map = load_json('data/books_map.json')
    return [books_map[str(i)] for i in id_recommendations]


if __name__ == "__main__":

    # load book features and user ratings
    _, V = load_features()
    user_ratings = load_json('user/ratings.json')

    # train and get user feature vector
    user_feature_vector = train_user_features(user_ratings, V, epochs=1000)

    # get recommendations and print it
    recommendations = get_recommendations(user_feature_vector, V, 10)
    print(recommendations)
