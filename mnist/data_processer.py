#!/usr/bin/env python
# -*- coding: utf-8 -*-

import input_data
import numpy as np

def load_image_data(path):
    """
    Args:
    path: string, file path of csv data file

    Returns:
    csv
    """

    import csv

    # load csv file
    with open(path, 'rb') as csv_file:
        csv_images = csv.reader(csv_file)
        images = np.array([d for d in csv_images])

    assert len(images[0]) == 784

    images = images.astype(np.float32)
    images = images / 255

    return images


def fix_images(mnist_data, my_data):
    """
    Args:
    mnist_data: [[float]] arary, mnist images feature vectors
    my_data: [[float]] arary, new my feature vectors
    n: int, number of add data

    Returns:
    numpy array
    """

    return np.r_[mnist_data, my_data]


def fix_labels(mnist_label, add_num):
    """
    Args:
    label: [[int]] arary, class labels
    n: int, number of add data

    Returns:
    [[int]] array

    """

    c_num = len(mnist_label[0])

    # add one dimention
    fixed_label = np.c_[mnist_label, np.zeros(len(mnist_label))]
    assert len(fixed_label[0]) == c_num + 1

    # generate new class label
    new_label = np.zeros(c_num + 1)
    new_label[c_num] = 1
    new_label = np.array([new_label for i in range(add_num)])

    # add new class label
    fixed_label = np.r_[fixed_label, new_label]
    assert len(fixed_label) == len(mnist_label) + add_num

    return fixed_label


def add_data(mnist_data, data_list):
    """
    Args:
    mnist_data: original mnist_data
    dat_list: numpy array list

    Return:
    numy array
    """

    data_num = len(data_list)

    train_images = mnist_data.train.images
    train_labels = mnist_data.train.labels
    test_images = mnist_data.test.images
    test_labels = mnist_data.test.labels

    for data in data_list:
        add_num = len(data)
        train_images = fix_images(train_images, data)
        train_labels = fix_labels(train_labels, add_num)
        test_images = fix_images(test_images, data)
        test_labels = fix_labels(test_labels, add_num)

    VALIDATION_SIZE = 5000

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    mnist_data.train = input_data.DataSet(
        train_images, train_labels, add=True)
    mnist_data.validation = input_data.DataSet(
        validation_images, validation_labels, add=True)
    mnist_data.test = input_data.DataSet(test_images, test_labels, add=True)

    return mnist_data


def main():
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    plus_data = load_image_data('csv/plus_data.csv')
    data = add_data(mnist_data, [plus_data])
    print data

if __name__ == "__main__":
    main()
