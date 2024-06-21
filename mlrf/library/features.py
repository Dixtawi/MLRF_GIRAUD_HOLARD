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

def hoc_features(images):
    """
    Exemple d'application:
    X_train_hoc = hoc_features(X_train)
    X_test_hoc = hoc_features(X_test)
    """
    all_histograms = []
    
    for image in images:
        if image.ndim == 3 and image.shape[2] == 3:
            b, g, r = cv2.split(image)
        elif image.ndim == 2:
            b = g = r = image
        else:
            raise ValueError("Unsupported image format. Expected 2D or 3-channel (BGR) image.")

        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

        hoc_features = np.concatenate((hist_b.flatten(), hist_g.flatten(), hist_r.flatten()))
        hoc_features /= np.sum(hoc_features)
        all_histograms.append(hoc_features)
    
    return np.array(all_histograms)
