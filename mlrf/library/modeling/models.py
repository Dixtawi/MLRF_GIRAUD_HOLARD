from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

# Dictionnaire contenant les modèles et leurs instances
models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42),
    'SGDClassifier': SGDClassifier(max_iter=1000, random_state=42),
    'SVM': SVC(random_state=42)
}

