from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from joblib import dump, load
import os
from config import MODELS_DIR


models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(multi_class='multinomial', max_iter=1000),
    'SVM': SVC()
}

def save_model(model, model_name):
    """
    Sauvegarde un modèle dans un fichier.

    Arguments:
    model -- Le modèle à sauvegarder.
    model_name -- Le nom du fichier de sauvegarde.

    Retourne:
    Le chemin complet du fichier de sauvegarde.
    """
    models_dir = MODELS_DIR / model_name
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'{model_name}.joblib')
    dump(model, model_path)
    return model_path

def load_model(model_name):
    models_dir = MODELS_DIR / model_name
    model_path = os.path.join(models_dir, f'{model_name}.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier {model_path} n'existe pas.")
    model = load(model_path)
    return model