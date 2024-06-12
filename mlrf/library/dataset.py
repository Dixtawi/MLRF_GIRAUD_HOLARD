import pickle
import sys
import os
import numpy as np

def load_batch(fpath):
        import pickle
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            data = d[b'data']
            labels = d[b'labels']
            return data, labels

def load_dataset(data_dir):
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

    return x_train, x_test, y_train, y_test

data_dir = "data/processed/cifar-10-batches-py"

X_train, X_test, y_train, y_test = load_dataset(data_dir)