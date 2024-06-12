import sys
import os


class ModelTraining:
    
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.model_fit = None

    def train(self):
        self.model_fit = self.model.fit(self.X_train, self.y_train)
        return self.model_fit


