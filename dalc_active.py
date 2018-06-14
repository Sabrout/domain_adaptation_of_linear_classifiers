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


def plot_model(dataset, model, fig_name, closest=list(), furthest=list()):
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

    # We plot the closest points to h_sep
    plt.scatter(dataset.X[closest][:, 0], dataset.X[closest][:, 1], color='yellow')
    plt.scatter(dataset.X[furthest][:, 0], dataset.X[furthest][:, 1], color='green')

    # Save and show figure
    plt.savefig('results/{}.png'.format(fig_name))
    plt.show()


def furthest_n(X_array, n, clf):
    # array of sample distances to the hyperplane
    dists = clf.decision_function(X_array)
    # absolute distance to hyperplane
    abs_dists = np.abs(dists)
    return np.sort(np.flip(abs_dists.argsort(), 0)[:n])


def closest_n(X_array, n, clf):
    # array of sample distances to the hyperplane
    dists = clf.decision_function(X_array)
    # absolute distance to hyperplane
    abs_dists = np.abs(dists)
    return np.sort(abs_dists.argsort()[:n])


def active_dalc(cost=50, iterations=5):
    # Reading Datasets (Source, Target, Test)
    source, target, test = setup.read_data()
    data = dataset.Dataset()
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
    data.X = np.asarray(X)
    data.Y = np.asarray(Y)

    # Linear Separator h_sep
    h_sep = svm.SVC(kernel='linear', C=1.0)
    h_sep.fit(data.X, data.Y)

    # DALC initial model with Moon dataset's optimal parameters
    dalc = Dalc(0.6309573650360107, 0.1258925348520279)
    kernel = Kernel('rbf', 1.2559431791305542)
    classifier = dalc.learn(source, target, kernel)

    # Capacity per iteration
    capacity = cost//(2*iterations)
    closest_samples = closest_n(X, capacity, h_sep)
    furthest_samples = furthest_n(X, capacity, h_sep)

    # Removing misclassified source samples by DALC

    # Closest Samples
    for j in range(0, len(closest_samples)):
        if classifier.predict([data.X[j]])*data.Y[closest_samples[j]] < 0:
            data.X = np.delete(data.X, closest_samples[j], 0)
            data.Y = np.delete(data.Y, closest_samples[j], 0)
            # Updating other indices
            for k in range(j, len(closest_samples)):
                closest_samples[k] -= 1
            for k in range(0, len(furthest_samples)):
                if furthest_samples[k] >= closest_samples[j]:
                    furthest_samples[k] -= 1
            # Deleting the sample
            closest_samples[j] = -len(data.X)
    # Clearing closest_samples
    closest_samples = closest_samples[closest_samples >= 0]

    # Furthest Samples
    for j in range(0, len(furthest_samples)):
        if classifier.predict([data.X[j]]) * data.Y[furthest_samples[j]] < 0:
            data.X = np.delete(data.X, furthest_samples[j], 0)
            data.Y = np.delete(data.Y, furthest_samples[j], 0)
            # Updating other indices
            for k in range(j, len(furthest_samples)):
                furthest_samples[k] -= 1
            for k in range(0, len(closest_samples)):
                if closest_samples[k] >= furthest_samples[j]:
                    closest_samples[k] -= 1
            # Deleting the sample
            furthest_samples[j] = -len(data.X)
    # Clearing furthest_samples
    furthest_samples = furthest_samples[furthest_samples >= 0]

    # Removing closest and furthest from target
    
    # ADD closest and furthest to source
    # Retrain DALC
    # Retrain h_sep

    plot_model(data, h_sep, 'linear_separator', closest_samples, furthest_samples)


def main():
    active_dalc()

    print("---------------------------------------------------")
    print("                    FINISHED")
    print("---------------------------------------------------")


if __name__ == "__main__":
    main()