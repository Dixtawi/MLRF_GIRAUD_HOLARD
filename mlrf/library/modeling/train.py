import sys
import os


class ModelTraining:
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.pred = None

    def get_predictions(self):
        self.pred = self.model.predict(self.X_test)
        return self.pred
    
    def get_accuracy(self):
        return accuracy_score(self.y_test, self.pred)

# Créer le modèle Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
clf_fitted = clf.fit(X_train, y_train)

