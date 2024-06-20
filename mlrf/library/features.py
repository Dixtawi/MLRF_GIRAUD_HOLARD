import cv2
import numpy as np
from skimage.feature import hog
from skimage import color

def flatten_images(images):
    """
    Exemple d'application:
    X_train_flat = flatten_images(X_train)
    X_test_flat = flatten_images(X_test)
    """
    flattened_images_list = []
    
    for image in images:
        flattened_images_list.append(image.flatten())
    
    flattened_images_array = np.array(flattened_images_list)
    
    return flattened_images_array


def hog_features(images):
    """
    Exemple d'application:
    X_train_hog = hog_features(X_train)
    X_test_hog = hog_features(X_test)
    """
    hog_features_list = []
    
    for image in images:
        image_gray = color.rgb2gray(image)
        features = hog(image_gray, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
        hog_features_list.append(features)
    
    hog_features_array = np.array(hog_features_list)
    
    return hog_features_array


def sift_features(images):
    sift = cv2.SIFT_create()
    all_keypoints = []
    all_descriptors = []
    
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)
    
    return all_keypoints, all_descriptors


def visualize_sift_features(images, sift_features_list, max_descriptors=500):
    """
    Visualise les points clés SIFT détectés sur l'image originale.

    Parameters:
    images (numpy array): Le groupe d'images original.
    sift_features_list (list): La liste des descripteurs SIFT aplatis.
    max_descriptors (int): Nombre maximum de descripteurs à utiliser pour chaque image.

    Returns:
    None
    """
    sift = cv2.SIFT_create()

    for i, image in enumerate(images[:5]):  # Limiter à 5 images pour visualisation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Limiter le nombre de points clés à max_descriptors
        if keypoints:
            keypoints = keypoints[:max_descriptors]

        # Dessiner les points clés sur l'image
        img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Afficher l'image avec les points clés
        plt.figure(figsize=(10, 10))
        plt.title(f"SIFT Features for Image {i+1}")
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

