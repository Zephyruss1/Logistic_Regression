import os
import numpy as np
import struct
from array import array
from sklearn import datasets

current_work_dir = os.path.dirname(__file__)
# MNIST veri kümesindeki 0 ve 1 rakamları bulunan veriler üzerinden işlem
# gerçekleşecektir.
target_ind_0 = 0
target_ind_1 = 1


class MnistDataloader(object):
    '''
    Load MNIST dataset
    '''

    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        return (x_train, y_train), (x_test, y_test)


def mnist_process_data(target_ind_0=0, target_ind_1=1):
    '''
    load dataset
    '''
    input_path = os.path.join(current_work_dir, './mnist/')
    training_images_filepath = os.path.join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels.idx1-ubyte')
    Dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                 test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = Dataloader.load_data()

    '''
    select target labels
    '''
    y_train_ind_0 = np.where(y_train == target_ind_0)
    y_train_ind_0 = y_train_ind_0[:3001]
    y_train_ind_1 = np.where(y_train == target_ind_1)
    y_train_ind_1 = y_train_ind_1[:3001]
    y_train_ind = np.union1d(y_train_ind_0, y_train_ind_1)
    x_train = x_train[y_train_ind]
    y_train = y_train[y_train_ind]

    y_test_ind_0 = np.where(y_test == target_ind_0)
    y_test_ind_1 = np.where(y_test == target_ind_1)
    y_test_ind = np.union1d(y_test_ind_0, y_test_ind_1)
    x_test = x_test[y_test_ind]
    y_test = y_test[y_test_ind]

    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    x_train = np.column_stack((x_train, np.ones((x_train.shape[0],))))
    x_test = np.column_stack((x_test, np.ones((x_test.shape[0],))))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalization
    x_train /= 255.0
    x_test /= 255.0

    return (x_train, y_train), (x_test, y_test)


def data_preprocess(args):
    '''
    return a tuple: (x_train, y_train), (x_test, y_test)
    each batch has shape: (n, d)
    - n: # of samples
    - d: sample dimension
    '''

    (x_train, y_train), (x_test, y_test) = mnist_process_data(target_ind_0=target_ind_0, target_ind_1=target_ind_1)

    print('==========================')
    print("     training dataset: ", x_train.shape)
    print("     testing dataset: ", x_test.shape)

    return (x_train, y_train), (x_test, y_test)
