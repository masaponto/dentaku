#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import input_data


def load_csv(path):
    """
    Args:
    path: string, file path of csv data file
    """

    import csv

    # load csv file
    with open(path, 'rb') as csv_file:
        data = csv.reader(csv_file)
        data_array = np.array([d for d in data])

    assert len(data_array[0]) == 784
    print len(data_array)

    return data_array


def fix_labels(label, n):
    """
    Args:
    label: [[int]] arary, class labels
    n: int, number of add data
    """

    # add one dimention
    fixed_label = np.c_[label, np.zeros(len(label))]
    assert len(fixed_label[0]) == len(label[0]) + 1

    # generate new class label
    new_label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    new_labels = np.array([new_label for i in range(n)])

    # add new class label
    fixed_label = np.r_[fixed_label, new_labels]
    assert len(fixed_label) == len(label) + n

    return fixed_label


def fix_images(mnist_data, my_data, n):
    """
    Args:
    mnist_data: [[float]] arary, mnist images feature vectors
    my_data: [[float]] arary, new my feature vectors
    n: int, number of add data
    """

    fixed_images = np.r_[mnist_data, my_data]

    return fixed_images


def add_data(mnist_data):

    VALIDATION_SIZE = 5000

    # load my dat from csv
    path = 'csv/plus_data.csv'
    my_data = load_csv(path)
    my_data_array = np.array(my_data).astype(np.float32)
    my_data_array = my_data_array / 255


    # add my data
    n = len(my_data_array)

    fixed_train_labels = fix_labels(mnist_data.train.labels, n)
    fixed_train_images = fix_images(
        mnist_data.train.images, my_data_array, n)

    fixed_test_labels = fix_labels(mnist_data.test.labels, n)
    fixed_test_images = fix_images(
        mnist_data.test.images, my_data_array, n)

    # split data
    fixed_validation_images = fixed_train_images[:VALIDATION_SIZE]
    fixed_validation_labels = fixed_train_labels[:VALIDATION_SIZE]
    fixed_train_images = fixed_train_images[VALIDATION_SIZE:]
    fixed_train_labels = fixed_train_labels[VALIDATION_SIZE:]

    #input_data.DataSet(mnist_data.tarin.tra ,fix_labels)

    mnist_data.train = input_data.DataSet(
        fixed_train_images, fixed_train_labels, add=True)
    mnist_data.validation = input_data.DataSet(
        fixed_validation_images, fixed_validation_labels, add=True)
    mnist_data.test = input_data.DataSet(fixed_test_images, fixed_test_labels, add=True)

    return mnist_data


def load_data():
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    return add_data(mnist_data)


if __name__ == "__main__":
    print load_data()
