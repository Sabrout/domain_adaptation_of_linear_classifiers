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


def get_source_data(data, labels):
    result = dataset.Dataset()
    result.X = list()
    result.Y = list()
    for i in range(0, len(data.Y)):
        if data.Y[i] == 1:
            result.X.append(data.X[i])
            result.Y.append(labels[i])
    # print(len(result.Y))
    return result


def get_target_data(data, labels):
    result = dataset.Dataset()
    result.X = list()
    result.Y = list()
    for i in range(0, len(data.Y)):
        if data.Y[i] == -1:
            result.X.append(data.X[i])
            result.Y.append(labels[i])
    # print(len(result.Y))
    return result


def filtering_samples(data, labels, classifier, closest_samples, furthest_samples):
    # Closest Samples
    for j in range(0, len(closest_samples)):
        if classifier.predict([data.X[j]]) * data.Y[closest_samples[j]] < 0 and data.Y[closest_samples[j]] == 1:
            data.X = np.delete(data.X, closest_samples[j], 0)
            data.Y = np.delete(data.Y, closest_samples[j], 0)
            labels = np.delete(labels, closest_samples[j], 0)
            # Updating other indices
            for k in range(j, len(closest_samples)):
                closest_samples[k] -= 1
            for k in range(0, len(furthest_samples)):
                if furthest_samples[k] >= closest_samples[j]:
                    furthest_samples[k] -= 1
            # Deleting the sample
            # print("deleted {} label {}".format(closest_samples[j], data.Y[closest_samples[j]]))
            closest_samples[j] = -len(data.X)
    # Clearing closest_samples
    closest_samples = closest_samples[closest_samples >= 0]

    # Furthest Samples
    for j in range(0, len(furthest_samples)):
        if classifier.predict([data.X[j]]) * data.Y[furthest_samples[j]] < 0 and data.Y[furthest_samples[j]] == 1:
            data.X = np.delete(data.X, furthest_samples[j], 0)
            data.Y = np.delete(data.Y, furthest_samples[j], 0)
            labels = np.delete(labels, furthest_samples[j], 0)
            # Updating other indices
            for k in range(j, len(furthest_samples)):
                furthest_samples[k] -= 1
            for k in range(0, len(closest_samples)):
                if closest_samples[k] >= furthest_samples[j]:
                    closest_samples[k] -= 1
            # Deleting the sample
            # print("deleted {} label {}".format(furthest_samples[j], data.Y[furthest_samples[j]]))
            furthest_samples[j] = -len(data.X)
    # Clearing furthest_samples
    furthest_samples = furthest_samples[furthest_samples >= 0]
    return closest_samples, furthest_samples, labels


def print_template(data, labels, closest_samples, furthest_samples):
    print(closest_samples)
    temp = list()
    for i in closest_samples:
        temp.append(data.Y[i])
    print(temp)
    print(furthest_samples)
    temp = list()
    for i in furthest_samples:
        temp.append(data.Y[i])
    print(temp)
    get_source_data(data, labels)
    get_target_data(data, labels)


def active_iteration(data, labels, dalc, classifier, kernel, h_sep, cost=50, iterations=5):

    # Capacity per iteration
    capacity = cost//(2*iterations)
    if capacity < 1:
        raise Exception('---- OUT OF COST ----')
    closest_samples = closest_n(data.X, capacity, h_sep)
    furthest_samples = furthest_n(data.X, capacity, h_sep)

    # Print Template
    # print_template(data, labels, closest_samples, furthest_samples)

    # Removing misclassified source samples by DALC
    closest_samples, furthest_samples, labels = filtering_samples(data, labels, classifier, closest_samples, furthest_samples)

    # Print Template
    # print_template(data, labels, closest_samples, furthest_samples)

    # Removing closest and furthest from target and adding them to source
    for j in closest_samples:
        if data.Y[j] == -1:
            if cost == 0:
                raise Exception('---- OUT OF COST ----')
            cost -= 1
        data.Y[j] = 1
    for j in furthest_samples:
        if data.Y[j] == -1:
            if cost == 0:
                raise Exception('---- OUT OF COST ----')
            cost -= 1
        data.Y[j] = 1

    # Retrain DALC
    classifier = dalc.learn(get_source_data(data, labels), get_target_data(data, labels), kernel)
    # Retrain h_sep
    h_sep.fit(data.X, data.Y)

    plot_model(data, h_sep, 'linear_separator', closest_samples, furthest_samples)

    return data, labels, classifier, h_sep, cost


def active_dalc(source, target, cost=50, iterations=5):

    data = dataset.Dataset()
    X = list()
    Y = list()
    labels = list()
    # Labeling Source Dataset
    for i in source.X:
        X.append(i)
        Y.append(1)
    # Labeling Target Dataset
    for i in target.X:
        X.append(i)
        Y.append(-1)
    # Saving DALC labels
    for i in source.Y:
        labels.append(i)
    for i in target.Y:
        labels.append(i)
    labels = np.asarray(labels)
    # Saving Labels in sep_dataset
    data.X = np.asarray(X)
    data.Y = np.asarray(Y)

    # Linear Separator h_sep
    h_sep = svm.SVC(kernel='linear', C=1.0)
    h_sep.fit(data.X, data.Y)

    # DALC initial model with Moon dataset's optimal parameters
    dalc = Dalc(0.6309573650360107, 0.1258925348520279)
    kernel = Kernel('rbf', 1.2559431791305542)
    classifier = dalc.learn(get_source_data(data, labels), get_target_data(data, labels), kernel)

    # Iteration Loop
    for i in range(0, iterations):
        data, labels, classifier, h_sep, cost = \
            active_iteration(data, labels, dalc, classifier, kernel, h_sep, cost, iterations)

    return dalc, classifier


def main():
    # Reading Datasets (Source, Target, Test)
    source, target, test = setup.read_data()

    # Executing Algorithm
    dalc, classifier = active_dalc(source, target, 50, 5)

    # Predictions
    predictions = classifier.predict(test.X)

    # Calculating Risk
    risk = classifier.calc_risk(test.Y, predictions=predictions)
    print('Test risk = ' + str(risk))

    print("---------------------------------------------------")
    print("                    FINISHED")
    print("---------------------------------------------------")


if __name__ == "__main__":
    main()