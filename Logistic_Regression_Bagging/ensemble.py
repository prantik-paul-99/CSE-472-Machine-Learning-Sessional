from data_handler import bagging_sampler
import numpy as np
import copy

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.bagging_models = []

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement

        for i in range(self.n_estimator):
            X_bagging, y_bagging = bagging_sampler(X, y)
            bagging_model = copy.deepcopy(self.base_estimator)
            bagging_model.fit(X_bagging, y_bagging)
            self.bagging_models.append(bagging_model)

        return self

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        
        y_pred = []

        for bagging_model in self.bagging_models:
            y_pred.append(bagging_model.predict(X))

        y_pred = np.array(y_pred)
        y_pred = np.sum(y_pred, axis=0)

        for i in range(len(y_pred)):
            if y_pred[i] >= self.n_estimator/2:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        return y_pred