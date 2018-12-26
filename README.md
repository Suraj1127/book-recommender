# Book Recommendation System

## Overview
Recommendation systems are one of the lucrative applications of machine learning.  Big companies have been able to
increase their sales and thereby generate more revenue by employing ML powered recommendation
engines in their applications.  Some companies employing different forms of recommendation systems are Amazon, Netflix, 
YouTube, Facebook, etc.

### Scope
We have developed recommendation system for books.  User rates few books and gives data in the form of 'json' file and 
the system prints the title of recommended books.

### Algorithms
We have just used collaborative filtering machine learning algorithm here.  The algorithm is written from scratch so
might not have efficient training time performance.

### Technologies
Everything from data loading to training is done using Python and its libraries.  The numeric computation is performed 
by numeric library of Python called NumPy.  Other libraries used for utilities functions are Pandas and Sklearn.

## Dependencies
Python 3 is should be used here as the interpreter.
* NumPy
* Sklearn
* Pandas

## Installation
Assuming Python 3 and pip3 is used,
* sudo pip3 install numpy
* sudo pip3 install pandas
* sudo pip3 install scikit-learn

## Usage
1. Provide your ratings to books in the [json](user/ratings.json) file.  The keys represent `id` for the books as given
in [books.csv](data/raw/books.csv) file and the values represent your ratings to the books.
2. Run the file `recommender.py`.
```
python3 recommender.py
```
You would get the `titles` of the recommended books.

## Developers Zone
This section helps developers to build their own custom application on top of this recommendation system.  Developers 
are encouraged to build other projects on top of this, improve different aspects of the existing repository or come up with cool 
features/ideas on this repository.  As the project is MIT licensed, the terms and conditions as given in LICENCE file should be followed.  In case of any other 
queries, the owner of this repository shall be contacted.

### Directories
* data: This directory contains two sub directories, `raw` and `processed`.  `Processed` directory was created to put
clean data but as the data itself was clean with no missing values, we decided to use the csv files in `raw` directory.
* data_exploration: Contains `exploration.ipynb` notebook to do data exploration.  Developers are encouraged to explore
the files under `data/raw` directory to have general understanding of data first.
* features: Contains matrix of features (both users and books).  The features are serialized to pickle format so
developers should unpickle it to use the features in any way.
* user: Contains `ratings.json` which contains ratings given by user (to whom recommendations are to be shown).  See 
[usage](#usage) for details.

### Modules
* _collaborative_filtering.py_: Contains collaborative filtering algorithm with evaluator function evaluating performance
on test set (2% of the whole dataset).  Root Mean Squared Error is taken as evaluation metric.
* _feature_generator.py_: Generates features matrices for books and users in pickle format.  The pickle files contain
deserialized NumPy arrays.
* _recommender.py*_: Recommends the user `title` of books by reading the `user/ratings.json` file.

### Performance Tips
* The number of latent features (K) is hyperparameter here which can be tuned for optimum performance.
* Updating of parameters can be done vectorically so as to improve training speed.
* Advanced optimization methods can be used to imply later epochs of trainings as finetuning epochs.
* We can use mini-batch gradient descent which reduces training time.
* SVD matrix factorization can be tested to see how it behaves by putting unrated elements of sparse matrix as 0.  
* Learning rate itself can be fine tuned and regularization can be used if the model overfits the training data.

## Data Source
https://www.kaggle.com/zygmunt/goodbooks-10k

About 2k rows in `ratings.csv` file are duplicated.  We can remove the duplicated rows as they don't provide any new 
information.

Data released under: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
## Licence
MIT License
