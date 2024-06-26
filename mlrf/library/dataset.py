import pickle
import sys
import os
import numpy as np
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_batch(fpath):
        import pickle
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            data = d[b'data']
            labels = d[b'labels']
            return data, labels

def load_dataset():
    data_dir = RAW_DATA_DIR / "cifar-10-batches-py"
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 32, 32, 3), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(data_dir, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(data_dir, 'test_batch')
    x_test, y_test = load_batch(fpath)
    x_test = x_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(y_test)
    
    # Normaliser les images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0


    return x_train, x_test, y_train, y_test

def save_dataset(dataset, name):
    pickle_file_path = PROCESSED_DATA_DIR / name + ".pickle"

    with open(pickle_file_path, 'wb') as f:
        pickle.dump(dataset, f)