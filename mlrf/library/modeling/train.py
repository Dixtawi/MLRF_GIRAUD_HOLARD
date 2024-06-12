import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'library')))

import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataset import X_train, X_test, y_train, y_test



# Créer le modèle Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
clf_fitted = clf.fit(X_train, y_train)

