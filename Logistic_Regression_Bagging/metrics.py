"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""
class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                TP += 1
            elif y_true[i] == 0 and y_pred[i] == 0:
                TN += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                FP += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                FN += 1

        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN

        print("True Positive: ", self.TP)
        print("True Negative: ", self.TN)
        print("False Positive: ", self.FP)
        print("False Negative: ", self.FN)

    def accuracy(self):

        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def precision_score(self):

        if(self.TP+self.FP == 0):
            return 0.0

        return self.TP / (self.TP + self.FP)

    def recall_score(self):

        if(self.TP+self.FN == 0):
            return 0.0

        return self.TP / (self.TP + self.FN)

    def f1_score(self):

        if(2*self.TP+self.FP+self.FN == 0):
            return 0.0

        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)
