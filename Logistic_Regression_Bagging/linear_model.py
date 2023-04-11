import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.learn_rt = params['learn_rt']
        self.num_itr = params['num_itr']
        self.loss_vals = []
        self.weight_vals = None
        self.bias_vals  = None

    def sigmoid(self, func_val):
        return 1 / (1 + np.exp(-func_val))

    def calculate_loss_val(self, X, y):

        func_val = np.dot(X, self.weight_vals) + self.bias_vals
        sigmoid_val = self.sigmoid(func_val)

        loss_y = y * np.log(sigmoid_val)
        loss_1_y = (1 - y) * np.log(1 - sigmoid_val)
        loss_val = - (loss_y + loss_1_y)
        loss_val = np.sum(loss_val) / len(X)

        return loss_val

    def gradient_calculation(self, X, y):

        func_val = np.dot(X, self.weight_vals) + self.bias_vals
        sigmoid_val = self.sigmoid(func_val)

        dw = np.dot(X.T, (sigmoid_val - y)) / len(X) + self.lambda_val * self.weight_vals / len(X)
        db = np.sum(sigmoid_val - y) / len(X)

        return dw, db

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

        no_of_weights = X.shape[1]
        self.weight_vals = np.zeros(no_of_weights)
        self.bias_vals = 0
        self.lambda_val = 0.01

        for i in range(self.num_itr):

            self.gradient_vals = self.gradient_calculation(X, y)
            self.weight_vals = self.weight_vals - self.learn_rt * self.gradient_vals[0]
            self.bias_vals = self.bias_vals - self.learn_rt * self.gradient_vals[1]

            loss_val = self.calculate_loss_val(X, y)
            self.loss_vals.append(loss_val)

            # if(i%100 == 0):
            #     print("Iteration: {} Loss: {}".format(i, loss_val))

        return self


    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement

        func_val = np.dot(X, self.weight_vals) + self.bias_vals
        sigmoid_val = self.sigmoid(func_val)

        no_smpl = len(X)

        y_pred = np.zeros(no_smpl)

        for i in range(no_smpl):
            if sigmoid_val[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        return y_pred
