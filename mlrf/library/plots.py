import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.base import clone
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_roc_curves(models, X_train, y_train, X_test, y_test):
    y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    
    plt.figure(figsize=(8, 6))
    
    for model_name, model in models.items():
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        if hasattr(model_clone, "predict_proba"):
            y_prob = model_clone.predict_proba(X_test)
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(y_test_bin.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        for i in range(y_test_bin.shape[1]):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'{model_name} (class {i} AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
    
    
def plot_confusion_matrice(model, model_name, X_train, y_train, X_test, y_test):
    
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    
def display_pca(x, y):
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(x.reshape((x.shape[0], -1)))

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_train_std)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=y, cmap='viridis', s=2)

    plt.colorbar(scatter, label='Classe')

    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.title('PCA des données d\'entraînement')
    plt.show()

def display_tsne(x, y):
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(x.reshape((x.shape[0], -1)))

    tsne = TSNE(n_components=2, random_state=42)
    X_train_tsne = tsne.fit_transform(X_train_std)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y, cmap='viridis', s=2)

    plt.colorbar(scatter, label='Classe')

    plt.xlabel('t-SNE Composante 1')
    plt.ylabel('t-SNE Composante 2')
    plt.title('Visualisation de l\'Espace Latent avec t-SNE')
    plt.show()