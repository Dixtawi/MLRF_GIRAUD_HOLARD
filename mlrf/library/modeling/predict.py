import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'library')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modeling')))

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from train import clf_fitted
from dataset import data_test, labels_test

y_pred = clf_fitted.predict(data_test)

accuracy = accuracy_score(labels_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
