import cv2

# Assurez-vous que les images sont en format 3D (nombre d'images, hauteur, largeur, canaux)
# Par exemple, si vos images sont en format (nombre d'images, hauteur * largeur * canaux), 
# vous pouvez les reshaper:
# num_images = images.shape[0]
# images = images.reshape(num_images, 32, 32, 3)  # Supposons des images 32x32x3

def extract_features(images):
    features = []
    for img in images:
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Redimensionner l'image (par exemple à 32x32 si nécessaire)
        gray = cv2.resize(gray, (32, 32))
        
        # Extraire les histogrammes de couleurs (3 canaux)
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        
        # Extraire les descripteurs de texture (LBP)
        lbp = local_binary_pattern(gray, 8, 1, method="uniform")
        (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        
        # Normaliser les histogrammes
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-6)
        
        # Combiner les features
        feature = np.hstack([hist, hist_lbp])
        
        features.append(feature)
    
    return np.array(features)