import numpy as np
import pandas as pd
import cv2
import os
import pickle
import PIL
from PIL import Image, ImageFilter
import os
import sys
import matplotlib.pyplot as plt
import time

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from train_1705071 import *

def main(path):

    cnn = Convolutional_neural_network()
    cnn.set_parameters(no_of_classes=10, learning_rate=0.001, samples_in_batch=16, epochs=20, in_channels = 1)
    cnn.build_architecture()
    cnn.load_model()

    # load the test data
    # test_data, test_labels, test_filenames = load_dataset_test(path = path, num_sample = 0, load_from_file = False, for_predict=False)
    test_data, test_labels, test_filenames = load_dataset_test(path = path, num_sample = 0, load_from_file = False, for_predict=True)

    # cnn.test(test_data, test_labels, test_filenames, path = path)

    cnn.predict(test_data, test_filenames, path = path)

if __name__ == '__main__':
    # if number of arguments is greater than 1, then the first argument is the path to the test data
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # path = 'Dataset/NumtaDB_with_aug/training-d'
        path = 'test-b1'

    main(path)