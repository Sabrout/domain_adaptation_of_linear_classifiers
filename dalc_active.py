#-*- coding:utf-8 -*-
'''
ACTIVE LEARNING USING DALC
See: https://github.com/Sabrout/domain_adaptation_of_linear_classifiers

Learning algorithm implementation

@author: Omar Elsabrout -- http://sabrout.github.io/
'''
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

def active_dalc(cost=50, iterations=10):
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    clf.fit(X, y)
    clf.predict([[2., 2.]])


def main():


    print("---------------------------------------------------")
    print("                    FINISHED")
    print("---------------------------------------------------")


if __name__ == "__main__":
    main()