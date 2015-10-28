#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" ELM

This script is ELM for mnist classification.
"""

__author__ = 'Masato'
__version__ = 1.0

import numpy as np

import pickle

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file


class ELM (BaseEstimator):

    """ ELM model Binary class classification
    """

    def __init__(self, hid_num, a=1):
        """
        Args:
        hid_num (int): number of hidden layer
        out_num (int): number of out layer
        a (int) : const value of sigmoid funcion

        """
        self.hid_num = hid_num
        self.a = a  # sigmoid constant value
        self.out_num = 10

    def _sigmoid(self, x):
        """sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid

        """

        return 1 / (1 + np.exp(- self.a * x))


    def fit(self, X, y):
        """ learning

        Args:
        X [[float]]: feature vectors of learnig data
        y [float] : labels of leanig data

        """

        x_vs = self._add_bias(X)

        np.random.seed()
        self.a_vs = np.random.uniform(-1.0, 1.0, (len(x_vs[0]), self.hid_num))
        self.out_num = max(y)  # number of class, number of output neuron

        h_t = np.linalg.pinv(self._sigmoid(np.dot(x_vs, self.a_vs)))

        if (self.out_num == 1):
            t_vs = y
            self.beta_v = np.dot(h_t, t_vs)

        else:
            t_vs = np.array(list(map(self._ltov(self.out_num), y)))
            self.beta_v = np.dot(h_t, t_vs)


    def dump_weights(self):
        print('dump')
        """dump weight to pickel file
        """

        weights = [self.a_vs, self.beta_v]
        fname = str(self.hid_num) + '_weights.dump'

        with open(fname, 'wb') as picfile:
            pickle.dump(weights, picfile)

    def load_weights(self):
        print('load')

        fname = 'elm/' + str(self.hid_num) + '_weights.dump'
        #fname = str(self.hid_num) + '_weights.dump'


        with open(fname, "rb") as f:
            self.a_vs, self.beta_v = pickle.load(f)

    def one_predict(self, x):
        return self.__vtol(np.sign(np.dot(self._sigmoid(np.dot(np.append(x,1), self.a_vs)), self.beta_v)))

    def _add_bias(self, x_vs):
        """add bias to list

        Args:
        vec [float]: vec to add bias

        Returns:
        [float]: added vec

        """

        return np.c_[x_vs, np.ones(len(x_vs))]

    def predict(self, X):
        """return classify result

        Args:
        X [[float]]: feature vectors of learnig data


        Returns:
        [int]: labels of classify result

        """

        return np.array(list(map(self.__vtol, np.sign(np.dot(self._sigmoid(np.dot(self._add_bias(X), self.a_vs)), self.beta_v)))))

    def __vtol(self, vec):
        """tranceform vector (list) to label

        Args:
        v: int list, list to transform

        Returns:
        int : label of classify result

        Exmples:
        >>> e = ELM(10, 3)
        >>> e._ELM__vtol([1, -1, -1])
        0
        >>> e._ELM__vtol([-1, 1, -1])
        1
        >>> e._ELM__vtol([-1, -1, 1])
        2
        >>> e._ELM__vtol([-1, -1, -1])
        0

        """

        if self.out_num == 1:
            return round(np.sign(vec), 0)
        else:
            v = list(vec)
            if len(v) == 1:
                return np.sign(vec[0])
            return v.index(max(v))

    def _ltov(self, n):
        """trasform label scalar to vector

        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label

        Exmples:
        >>> e = ELM(10, 3)
        >>> e._ltov(3)(0)
        [1, -1, -1, -1]
        >>> e._ltov(3)(1)
        [-1, 1, -1, -1]
        >>> e._ltov(3)(2)
        [-1, -1, 1, -1]
        >>> e._ltov(3)(3)
        [-1, -1, -1, 1]

        """
        def inltov(label):
            return [-1 if i != label else 1 for i in range(0, int(n) + 1)]
        return inltov


def check_classification():

    db_name = 'MNIST original'
    data_set = fetch_mldata(db_name)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.4, random_state=0)

    hid_num = 1000

    print(hid_num)
    e = ELM(hid_num)
    e.fit(X_train, y_train)
    #e.load_weights()
    re = np.array([ e.one_predict(x) for x in X_test])
    print(sum([r == y for r, y in zip(re, y_test)]) / len(y_test))



def run_iris_cv():
    db_name = 'iris'
    print(db_name)

    data_set = fetch_mldata(db_name)
    #data_set.data = data_set.data.astype(np.float32)
    #data_set.data /= 255     # 0-1のデータに変換
    #data_set.target = data_set.target.astype(np.int32)

    hid_num = 10
    print(hid_num)
    print(data_set.data.shape)

    e = ELM(hid_num)
    ave = 0

    for i in range(10):
        scores = cross_validation.cross_val_score(
            e, data_set.data, data_set.target, cv=10, scoring='accuracy')
        ave += scores.mean()

    ave /= 10
    print("Accuracy: %0.3f" % (ave))



def run_cv():

    db_name = 'MNIST original'
    print(db_name)

    data_set = fetch_mldata(db_name)
    data_set.data = data_set.data.astype(np.float32)
    data_set.data /= 255     # 0-1のデータに変換
    data_set.target = data_set.target.astype(np.int32)

    hid_num = 1000
    print(hid_num)
    print(data_set.data.shape)

    e = ELM(hid_num)
    ave = 0

    for i in range(3):
        scores = cross_validation.cross_val_score(
            e, data_set.data, data_set.target, cv=10, scoring='accuracy', n_jobs = -1)
        ave += scores.mean()

    ave /= 3
    print("Accuracy: %0.3f" % (ave))



def learn_elm():

    db_name = 'MNIST original'
    data_set = fetch_mldata(db_name)
    data_set.data = data_set.data.astype(np.float32)
    data_set.data /= 255     # 0-1のデータに変換
    data_set.target = data_set.target.astype(np.int32)

    e = ELM(1000)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.1, random_state=0)

    e.fit(X_train, y_train)
    e.dump_weights()


def main():
    check_classification()
    #run_cv()
    #run_iris_cv()
    #learn_elm()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
