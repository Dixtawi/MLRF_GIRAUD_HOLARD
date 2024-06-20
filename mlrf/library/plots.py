import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.base import clone
import numpy as np


def plot_roc_curves(models, X_train, y_train, X_test, y_test):
    y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    
    for model_name, model in models.items():
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        y_prob = model_clone.predict_proba(X_test)
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(y_test_bin.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        for i in range(y_test_bin.shape[1]):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'{model_name} (class {i} AUC = {roc_auc[i]:.2f})')
    
    plt.show()