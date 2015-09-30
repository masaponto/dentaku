#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" ELM

This script is ELM for binary and multiclass classification.
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

    def __sigmoid(self, x):
        """sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid

        """

        #return 1 / (1 + np.exp(- self.a * x))
        return 0.0 if self.a * x < -709 else 1 / (1 + np.exp(- self.a * x))

    def __G(self, a_v, x_v):
        """output hidden nodes

        Args:
        a_v ([float]): weight vector of hidden layer
        x_v ([float]): input vector

        Returns:
        float: output hidden nodes

        """

        return self.__sigmoid(np.dot(a_v, x_v))

    def __f(self, x_v):
        """output of NN
        Args:
        x_v ([float]): input vector

        Returns:
        int: labels

        """

        return np.dot(self.beta_v, [self.__G(a_v, x_v) for a_v in self.a_vs])
        #return np.sign(np.dot(self.beta_v, [self.__G(a_v, x_v) for a_v in self.a_vs]))

    def __get_hid_matrix(self, x_vs):
        """ output matrix hidden layer
        Args:
        x_vs ([[float]]): input vector

        Returns:
        [[float]]: output matrix of hidden layer

        """

        return np.array([[self.__G(a_v, x_v) for a_v in self.a_vs] for x_v in x_vs])


    def fit(self, X, y):
        """ learning

        Args:
        X [[float]]: feature vectors of learnig data
        y [float] : labels of leanig data

        """

        x_vs = np.array(list(map(self.__add_bias, X)))

        # weight hid layer
        self.a_vs = np.random.uniform(-1.0, 1.0, (self.hid_num, len(x_vs[0])))

        # output matrix hidden nodes
        h = self.__get_hid_matrix(x_vs)

        # pseudo-inverse matrix of H
        h_t = np.linalg.pinv(h)

        self.out_num = max(y)  # number of class, number of output neuron

        if (self.out_num == 1):
            t_vs = y
            # weight out layer
            self.beta_v = np.dot(h_t, t_vs)
            #del t_vs

        else:
            t_vs = np.array(list(map(self.__ltov(self.out_num), y)))
            # weight out layer
            self.beta_v = np.transpose(np.dot(h_t, t_vs))
            #del t_vs

        #del x_vs
        #del h
        #del h_t
        #gc.collect()


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

        fname = 'elm/' +str(self.hid_num) + '_weights.dump'

        with open(fname, "rb") as f:
            self.a_vs, self.beta_v = pickle.load(f)


    def one_predict(self, x):
        x = np.array(self.__add_bias(x))
        return self.__vtol(self.__f(x))


    def __add_bias(self, vec):
        """add bias to list

        Args:
        vec [float]: vec to add bias

        Returns:
        [float]: added vec

        """

        return np.append(vec, 1)

    def predict(self, X):
        """return classify result

        Args:
        X [[float]]: feature vectors of learnig data


        Returns:
        [int]: labels of classify result

        """

        X = np.array(list(map(self.__add_bias, X)))
        return np.array([self.__vtol(self.__f(xs)) for xs in X])


    def __vtol(self, vec):
        """tranceform vector (list) to label

        Args:
        v: int list, list to transform

        Returns:
        int : label of classify result

        Exmples:
        >>> e = MultiClassELM(10, 3, 1)
        >>> e._MultiClassELM__vtol([1, -1, -1])
        0
        >>> e._MultiClassELM__vtol([-1, 1, -1])
        1
        >>> e._MultiClassELM__vtol([-1, -1, 1])
        2
        >>> e._MultiClassELM__vtol([-1, -1, -1])
        0

        """

        if self.out_num == 1:
            return round(np.sign(vec), 0)
        else:
            v = list(vec)
            if len(v) == 1:
                return np.sign(vec[0])
            #return int(v.index(1))
            return v.index(max(v))


    def __ltov(self, n):
        """trasform label scalar to vector

            Args:
            n (int) : number of class, number of out layer neuron
            label (int) : label

            Exmples:
            >>> e = MultiClassELM(10, 3, 1)
            >>> e._MultiClassELM__ltov(3)(1)
            [-1, 1, -1, -1]
            >>> e._MultiClassELM__ltov(3)(2)
            [-1, -1, 1, -1]
            >>> e._MultiClassELM__ltov(3)(3)
            [-1, -1, -1, 1]

            """
        def _ltov(label):
            return [-1 if i != label else 1 for i in range(0, int(n) + 1)]
        return _ltov


def check_classification():

    db_name = 'MNIST original'
    data_set = fetch_mldata(db_name)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.4, random_state=0)

    hid_nums = range(500, 1100, 100)

    for hid_num in hid_nums:
        e = ELM(hid_num)
        e.fit(X_train, y_train)

        re = e.predict(X_test)
        print( sum([r == y for r, y in zip(re, y_test)]) / len(y_test))


def run_cv():
<<<<<<< HEAD
    db_name = 'australian'
    print(db_name)
    #db_name = 'MNIST original'
=======
    db_name = 'MNIST original'
    #db_name = 'iris'
    print(db_name)

>>>>>>> develop
    data_set = fetch_mldata(db_name)
    hid_num = 1000
    print(hid_num)
    print(data_set.data.shape)


    e = ELM(hid_num)
    ave = 0

    for i in range(10):
        scores = cross_validation.cross_val_score(
            e, data_set.data, data_set.target, cv=10, scoring='accuracy', n_jobs = -1)
        ave += scores.mean()

    ave /= 10
    print("Accuracy: %0.3f " % (ave))


def learn_elm():

    db_name = 'MNIST original'
    data_set = fetch_mldata(db_name)

    e = ELM(1000)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.2, random_state=0)

    e.fit(X_train, y_train)
    e.dump_weights()


def main():
    #run_cv()
    learn_elm()

if __name__ == "__main__":
    main()
