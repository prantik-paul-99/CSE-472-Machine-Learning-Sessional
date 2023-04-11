"""
main code that you will run
"""

from data_handler import load_dataset, split_dataset
from linear_model import LogisticRegression
from metrics import Metrics
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # data load
    X, y = load_dataset('data_banknote_authentication.csv')
    #X, y = load_dataset('airlines.csv')
    
    # split train and test

    test_size = float(input('Enter the test size: '))
    is_shuffle = input('Shuffle the data? (y/n): ')

    shuffle = False

    if is_shuffle == 'y':
        shuffle = True

    X_train, y_train, X_test, y_test = split_dataset(X, y, test_size, shuffle)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # training

    params = {}
    params['learn_rt'] = float(input('Enter the learning rate: '))
    params['num_itr'] = int(input('Enter the number of iterations: '))
    classifier = LogisticRegression(params)
    classifier.fit(X_train, y_train)

    #print losses
    print("loss in iteration 1 : %f" %(classifier.loss_vals[0]))
    print("loss in iteration %i : %f" %((int)(params['num_itr']/2), classifier.loss_vals[(int)(params['num_itr']/2)-1]))
    print("loss in iteration %i : %f" %(params['num_itr'], classifier.loss_vals[(int)(params['num_itr'])-1]))

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set

    metrics = Metrics(y_true=y_test, y_pred=y_pred)

    print('Accuracy ', metrics.accuracy())
    print('Precision score ', metrics.precision_score())
    print('Recall score ', metrics.recall_score())
    print('F1 score ', metrics.f1_score())