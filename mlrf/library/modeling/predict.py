import sys
import os

from sklearn.metrics import accuracy_score

class ModelPerformances:
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.pred = None

    def get_predictions(self):
        self.pred = self.model.predict(self.X_test)
        return self.pred
    
    def get_accuracy(self):
        self.pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, self.pred)
        