import numpy as np
import pandas as pd

def load_dataset(dataset_name):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement

    banknote_df = pd.read_csv(dataset_name)

    X = banknote_df.iloc[:, :-1].values
    y = banknote_df.iloc[:, -1].values

    # scale the features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.

    banknote_df = pd.DataFrame(data = np.column_stack((X, y)))

    if(shuffle):
        banknote_df = banknote_df.sample(frac=1)

    train_size = 1 - test_size

    X_train = banknote_df.iloc[:int(train_size * len(X)), :-1].values
    y_train = banknote_df.iloc[:int(train_size * len(X)), -1].values

    X_test = banknote_df.iloc[int(train_size * len(X)):, :-1].values
    y_test = banknote_df.iloc[int(train_size * len(X)):, -1].values

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    banknote_df = pd.DataFrame(data = np.column_stack((X, y)))
    
    banknote_df = banknote_df.sample(frac=1, replace=True)
    
    X_sample = banknote_df.iloc[:, :-1].values
    y_sample = banknote_df.iloc[:, -1].values

    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape

    return X_sample, y_sample
