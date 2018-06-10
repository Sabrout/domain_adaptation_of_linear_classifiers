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
from dalc import Dalc
from dalc import Kernel


def plot_model(dataset, model):
    # We will now create a 2d graphic to illustrate the learning result
    # We create a mesh to plot in
    h = .02  # grid step
    x_min = dataset.X[:, 0].min() - 1
    x_max = dataset.X[:, 0].max() + 1
    y_min = dataset.X[:, 1].min() - 1
    y_max = dataset.X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # The grid is created, the intersections are in xx and yy


    Z2d = model.predict(np.c_[xx.ravel(), yy.ravel()])  # We predict all the grid
    Z2d = Z2d.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z2d, cmap=plt.cm.Paired)
    # We plot also the training points
    plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.Y, cmap=plt.cm.coolwarm)
    plt.savefig('results/linear_separator.png')
    plt.show()


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

    # Linear Separator h_sep
    h_sep = svm.SVC(kernel='linear', C=1.0)
    h_sep.fit(sep_dataset.X, sep_dataset.Y)

    # DALC initial model with Moon dataset's optimal parameters
    dalc = Dalc(0.19952623546123505, 0.25118863582611084)
    kernel = Kernel('rbf', 1.2559431791305542)
    classifier = dalc.learn(source, target, kernel)

    # Capacity per iteration
    capacity = cost/(2*iterations)
    


    plot_model(sep_dataset, h_sep)


def main():
    active_dalc()

    print("---------------------------------------------------")
    print("                    FINISHED")
    print("---------------------------------------------------")


if __name__ == "__main__":
    main()