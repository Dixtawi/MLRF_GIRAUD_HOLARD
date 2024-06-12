import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'library')))

import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern

from dataset import data, labels


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Créer le modèle Random Forest
clf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2]
}

# Configurer la recherche en grille
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Entraîner la recherche en grille
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres
print(f'Best parameters found: {grid_search.best_params_}')

# Utiliser le meilleur modèle pour prédire les étiquettes pour l'ensemble de test
best_clf = grid_search.best_estimator_