#-*- coding:utf-8 -*-
'''
ACTIVE LEARNING USING DALC
See: https://github.com/Sabrout/domain_adaptation_of_linear_classifiers

Learning algorithm implementation

@author: Omar Elsabrout -- http://sabrout.github.io/
'''
from sklearn import svm
import dataset
import numpy as np
import setup
import matplotlib.pyplot as plt


def active_dalc(cost=50, iterations=10):
    # Reading Datasets (Source, Target, Test)
    source, target, test = setup.read_data()
    sep_dataset = dataset.Dataset()
    X = list()
    Y = list()
    # Labeling Source Dataset
    for i in source.X:
        X.append(i)
        Y.append(1)
    # Labeling Target Dataset
    for i in target.X:
        X.append(i)
        Y.append(-1)
    # Saving Labels in sep_dataset
    sep_dataset.X = np.asarray(X)
    sep_dataset.Y = np.asarray(Y)

    # We will now create a 2d graphic to illustrate the learning result
    # We create a mesh to plot in
    h = .02  # grid step
    x_min = sep_dataset.X[:, 0].min() - 1
    x_max = sep_dataset.X[:, 0].max() + 1
    y_min = sep_dataset.X[:, 1].min() - 1
    y_max = sep_dataset.X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # The grid is created, the intersections are in xx and yy

    # Linear Separator h_sep
    h_sep = svm.SVC(kernel='linear', C=1.0)
    h_sep.fit(sep_dataset.X, sep_dataset.Y)
    Z2d = h_sep.predict(np.c_[xx.ravel(), yy.ravel()])  # We predict all the grid
    Z2d = Z2d.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z2d, cmap=plt.cm.Paired)
    # We plot also the training points
    plt.scatter(sep_dataset.X[:, 0], sep_dataset.X[:, 1], c=sep_dataset.Y, cmap=plt.cm.coolwarm)
    plt.savefig('results/linear_separator.png')
    plt.show()


def main():
    active_dalc()

    print("---------------------------------------------------")
    print("                    FINISHED")
    print("---------------------------------------------------")


if __name__ == "__main__":
    main()