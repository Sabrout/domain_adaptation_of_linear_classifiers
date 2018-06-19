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
import pickle
from sklearn import datasets


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
    if fig_name == 'linear_seperator':
        plt.pcolormesh(xx, yy, Z2d, cmap=plt.cm.Paired)
    else:
        plt.pcolormesh(xx, yy, Z2d)
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


def add_point(data, X, Y):
    np.append(data.X, X)
    np.append(data.Y, Y)


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
            for i in range(0, 10):
                add_point(data, data.X[j], data.Y[j])
        data.Y[j] = 1
    for j in furthest_samples:
        if data.Y[j] == -1:
            if cost == 0:
                raise Exception('---- OUT OF COST ----')
            cost -= 1
            for i in range(0, 10):
                add_point(data, data.X[j], data.Y[j])
        data.Y[j] = 1

    # Retrain DALC
    classifier = dalc.learn(get_source_data(data, labels), get_target_data(data, labels), kernel)
    # Retrain h_sep
    h_sep.fit(data.X, data.Y)

    # plot_model(data, classifier, 'active_dalc', closest_samples, furthest_samples)

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
    data.X = X
    data.Y = Y

    # Linear Separator h_sep
    h_sep = svm.SVC(kernel='rbf', C=15.1)                # Manually Tuned Parameters
    h_sep.fit(data.X, data.Y)
    print('Separator Score = ' + str(h_sep.score(data.X, data.Y)))

    # DALC initial model with Moon dataset's optimal parameters
    dalc = Dalc(0.6309573650360107, 0.1258925348520279)  # Manually Tuned Parameters
    kernel = Kernel('rbf', 1.2559431791305542)           # Manually Tuned Parameters
    classifier = dalc.learn(get_source_data(data, labels), get_target_data(data, labels), kernel)

    # Iteration Loop
    for i in range(0, iterations):
        data, labels, classifier, h_sep, cost = \
            active_iteration(data, labels, dalc, classifier, kernel, h_sep, cost, iterations)

    return dalc, classifier, data, labels


def save_model(classifier, cost, iterations):
    filename = 'active\models\model-{}-{}.bin'.format(cost, iterations)
    try:
        with open(filename, 'wb') as model:
            pickle.dump(classifier, model, pickle.HIGHEST_PROTOCOL)
        print('File "' + filename + '" created.')
    except:
        print('ERROR: Unable to write model file "' + filename + '".')


def save_data(data, labels):
    # Saving source dataset
    datasets.dump_svmlight_file(get_source_data(data, labels).X, get_source_data(data, labels).Y,
                                'active\data\\source.svmlight', zero_based=True
                                , comment=None, query_id=None, multilabel=False)
    # Saving target dataset
    datasets.dump_svmlight_file(get_target_data(data, labels).X, get_target_data(data, labels).Y,
                                'active\data\\target.svmlight', zero_based=True
                                , comment=None, query_id=None, multilabel=False)


# def test_validate(cost, iteration):
#     # Testing the model
#     command = ""
#     command += "python dalc_classify.py -f svmlight -m active\models\model-{}-{}.bin " \
#                "-p active\predictions\predict-{}-{}.bin active\data\\test.svmlight >> ".format(cost, iteration,
#                                                                                          cost, iteration)
#     command += "active\\results\\classification_risk.txt"
#     os.system(command)
#     # Validating the model
#     command = ""
#     command += "python dalc_reverse_cv.py -b {} -c {} -k rbf -g {} -f svmlight " \
#                "data\source.svmlight data\\target.svmlight >> results\Validation.txt".format(i, j, k)
#     os.system(command)


def main():
    # Reading Datasets (Source, Target, Test)
    source, target, test = setup.read_data()

    # Executing Algorithm
    cost = 100
    iterations = 5
    dalc, classifier, data, labels = active_dalc(source, target, cost, iterations)
    save_data(data, labels)
    # save_model(classifier, cost, iterations)

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