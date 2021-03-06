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
import re
import glob


def plot_model(X, y, model, X_target, fig_name):
    h = .02  # step size in the mesh

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z2d = model.predict(np.c_[xx.ravel(), yy.ravel()])  # We predict all the grid
    Z2d = Z2d.reshape(xx.shape)
    plt.contourf(xx, yy, Z2d, levels=[-5, -0.01, 0.01, 5], cmap=plt.cm.coolwarm, alpha=0.5)
    plt.contour(xx, yy, Z2d,1, linewidths=2.0, colors='black')

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.scatter(X_target[:, 0], X_target[:, 1], c='black', marker='.')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('Cost = 25% - Iteration No.{}'.format(int(re.search(r'\d+', fig_name).group()) + 1))
    plt.savefig('results/model_plots/{}.png'.format(fig_name))
    # plt.show()
    plt.close()


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


def is_included(data, X, Y):
    if data.X == None:
        return False
    for i in range(0, len(data.X)):
        if data.X[i] == X and data.Y[i] == Y:
            return True
        else:
            return False


def filtering_samples(data, labels, target_cache, classifier, closest_samples, furthest_samples):
    # Closest Samples
    for j in range(0, len(closest_samples)):
        if classifier.predict([data.X[j]]) * data.Y[closest_samples[j]] < 0 and data.Y[closest_samples[j]] == 1\
                and not is_included(target_cache, data.X[closest_samples[j]], data.Y[closest_samples[j]]) \
                and data.Y[furthest_samples[j]] == 1:
            # data.X = np.delete(data.X, closest_samples[j], 0)
            # data.Y = np.delete(data.Y, closest_samples[j], 0)
            # labels = np.delete(labels, closest_samples[j], 0)
            # # Updating other indices
            # for k in range(j, len(closest_samples)):
            #     closest_samples[k] -= 1
            # for k in range(0, len(furthest_samples)):
            #     if furthest_samples[k] >= closest_samples[j]:
            #         furthest_samples[k] -= 1
            # # Deleting the sample
            # # print("deleted {} label {}".format(closest_samples[j], data.Y[closest_samples[j]]))
            # closest_samples[j] = -len(data.X)
            continue
    # Clearing closest_samples
    closest_samples = closest_samples[closest_samples >= 0]

    # Furthest Samples
    for j in range(0, len(furthest_samples)):
        if classifier.predict([data.X[j]]) * data.Y[furthest_samples[j]] < 0 and data.Y[furthest_samples[j]] == 1 \
                and not is_included(target_cache, data.X[furthest_samples[j]], data.Y[furthest_samples[j]])\
                and data.Y[furthest_samples[j]] == 1:
            # data.X = np.delete(data.X, furthest_samples[j], 0)
            # data.Y = np.delete(data.Y, furthest_samples[j], 0)
            # labels = np.delete(labels, furthest_samples[j], 0)
            # # Updating other indices
            # for k in range(j, len(furthest_samples)):
            #     furthest_samples[k] -= 1
            # for k in range(0, len(closest_samples)):
            #     if closest_samples[k] >= furthest_samples[j]:
            #         closest_samples[k] -= 1
            # # Deleting the sample
            # # print("deleted {} label {}".format(furthest_samples[j], data.Y[furthest_samples[j]]))
            # furthest_samples[j] = -len(data.X)
            continue
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


def active_iteration(data, target, test, labels, target_cache,
                     dalc, classifier, kernel, h_sep, cost=50, iteration_number=0, fig_name=''):

    if cost < 1:
        # print("COST = {}".format(cost))
        raise Exception('---- OUT OF COST ----')

    # Print Template
    # print_template(data, labels, closest_samples, furthest_samples)
    capacity_close = cost // 2
    capacity_far = cost // 2
    while cost > 0:
        tmp_cost = cost
        break_bool = False
        # print('cost {}'.format(cost))
        # print('-----------')

        closest_samples = closest_n(data.X, capacity_close, h_sep)
        furthest_samples = furthest_n(data.X, capacity_far, h_sep)
        # Removing misclassified source samples by DALC
        closest_samples, furthest_samples, labels = filtering_samples(data, labels, target_cache, classifier
                                                                      , closest_samples, furthest_samples)

        # Print Template
        # print_template(data, labels, closest_samples, furthest_samples)

        # Removing closest and furthest from target and adding them to source
        for j in closest_samples:
            if data.Y[j] == -1:
                if cost == 0:
                    break_bool = True
                    break
                cost -= 1
                # Adding to target_cache
                add_point(target_cache, data.X[j], data.Y[j])
                # Keeping in Target
                add_point(data, data.X[j], -1)
                # for i in range(0, 10):
                #     add_point(data, data.X[j], data.Y[j])
            data.Y[j] = 1
        for j in furthest_samples:
            if break_bool: break
            if data.Y[j] == -1:
                if cost == 0:
                    break_bool = True
                    break
                cost -= 1
                # Adding to target_cache
                add_point(target_cache, data.X[j], data.Y[j])
                # Keeping in Target
                add_point(data, data.X[j], -1)
                # for i in range(0, 10):
                #     add_point(data, data.X[j], data.Y[j])
            data.Y[j] = 1

        if break_bool:
            break

        if cost == tmp_cost:
            # capacity_close += 1
            capacity_far += 1

    # Retrain DALC
    classifier = dalc.learn(get_source_data(data, labels), get_target_data(data, labels), kernel)
    # Retrain h_sep
    h_sep.fit(data.X, data.Y)

    # plot_model(get_source_data(data, labels).X, get_source_data(data, labels).Y, classifier, test.X,
    #            'ItrByItr/Iteration{}'.format(iteration_number))

    return data, labels, classifier, h_sep, cost


def active_dalc(source, target, test, cost=50, iterations=5, B=1.0, C=1.0, G=1.0, fig_name=''):

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
    h_sep = svm.SVC(kernel='rbf', C=1.0)                # Manually Tuned Parameters
    h_sep.fit(data.X, data.Y)
    # print('Separator Score = ' + str(h_sep.score(data.X, data.Y)))

    # DALC initial model with Moon dataset's optimal parameters
    dalc = Dalc(B, C)
    kernel = Kernel('rbf', G)
    classifier = dalc.learn(get_source_data(data, labels), get_target_data(data, labels), kernel)
    classic_dalc = classifier

    # Capacity per iteration
    capacity = cost // (2 * iterations)

    # target_cache for previously selected points in Target
    target_cache = dataset.Dataset()
    # Iteration Loop
    for i in range(0, iterations):
        data, labels, classifier, h_sep, cost = \
            active_iteration(data, target, test
                             , labels, target_cache, dalc, classifier, kernel, h_sep, capacity,
                             i, fig_name)
        plot_model(source.X, source.Y, classifier, test.X, 'ItrByItr/Iteration{}'.format(i))

    # # Plotting
    # plot_model(source.X, source.Y, classifier, test.X, fig_name)
    print('.')
    return classic_dalc, classifier, data, labels


def save_model(classifier, cost, iterations):
    filename = 'active\models\model-{}-{}.bin'.format(cost, iterations)
    try:
        with open(filename, 'wb') as model:
            pickle.dump(classifier, model, pickle.HIGHEST_PROTOCOL)
        print('File "' + filename + '" created.')
    except:
        print('ERROR: Unable to write model file "' + filename + '".')


def save_data(data, labels, rotation=''):
    # Saving source dataset
    datasets.dump_svmlight_file(get_source_data(data, labels).X, get_source_data(data, labels).Y,
                                'active\data\\source{}.svmlight'.format(rotation), zero_based=True
                                , comment=None, query_id=None, multilabel=False)
    # Saving target dataset
    datasets.dump_svmlight_file(get_target_data(data, labels).X, get_target_data(data, labels).Y,
                                'active\data\\target{}.svmlight'.format(rotation), zero_based=True
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


def multiple_rotations_experiment(cost, iterations, start_angle, end_angle):
    for i in range(start_angle, end_angle, 10):
        # Generating Datasets
        datasets = setup.generate_moon_dataset(200, 200, 1000, i)
        # Reading Datasets (Source, Target, Test)
        source, target, test = setup.read_data()

        setup.plot_datasets(datasets, 'rotation{}'.format(i))
        # setup.dalc_tune(0.1, 0.2, 0.1, 0.2, 0.5, 0.5)
        setup.dalc_tune(0.1, 1.2, 0.1, 1.2, 0.1, 0.1)
        model = setup.extract_model()
        text_file = open("results\\optimal_models.txt", "a")
        text_file.write("---------------------------------------------------\n")
        text_file.write("OPTIMAL MODEL FOR MOON DATASET (ROTATION = {})\n".format(i))
        text_file.write("CASE : B = {}, C = {}, Gamma = {}\n".format(model[0], model[1], model[2]))
        text_file.write("Validation Risk = {}\n".format(model[3]))
        text_file.write("Standard Deviation = {}\n".format(model[5]))
        text_file.write("Classification Risk = {}\n".format(model[4]))
        text_file.write("---------------------------------------------------\n")
        text_file.close()

        # Executing Algorithm
        dalc, classifier, data, labels = active_dalc(source, target, test, cost, iterations, model[0], model[1], model[2]
                                                     , 'moon_dataset_rotation{}'.format(i))
        save_data(data, labels, str(i))

        # Predictions for Classic DALC
        predictions_dalc = dalc.predict(test.X)
        # Calculating Risk for Classic DALC
        risk_dalc = dalc.calc_risk(test.Y, predictions=predictions_dalc)

        # Predictions for Active DALC
        predictions = classifier.predict(test.X)
        # Calculating Risk for Active DALC
        risk = classifier.calc_risk(test.Y, predictions=predictions)

        # print('ROTATION({}) >> Test risk = '.format(i) + str(risk))
        # print('--------------------------------------------------')

        text_file = open("results\\empirical_results.txt", "a")
        text_file.write("===================================================\n")
        text_file.write("MOON DATASET (ROTATION ={})\n".format(i))
        text_file.write("ACTIVE DALC\n")
        text_file.write("Classification Risk = {}\n".format(str(risk)))
        text_file.write("---------------------------------------------------\n")
        text_file.write("CLASSIC DALC\n")
        text_file.write("Classification Risk = {}\n".format(str(risk_dalc)))
        text_file.write("===================================================\n")
        text_file.close()

        # Plotting
        plot_model(source.X, source.Y, classifier, test.X, 'rotation{}'.format(i))


def multiple_cost_experiment(start_cost, end_cost):
    for i in range(start_cost, end_cost, 10):
        result = list()
        classic_result = list()
        for j in range(0, 6):
            datasets = setup.generate_moon_dataset_save_all(200, 200, 1000, 90, i, j)
            # Reading Datasets (Source, Target, Test)
            source, target, test = setup.read_all_data(i, j)

            # Executing Algorithm
            dalc, classifier, data, labels = active_dalc(source, target, test, i, 5
                                                         , 1.0, 1.0, 0.5
                                                         , str(70))
            # save_data(data, labels, str(i))

            # Predictions for Classic DALC
            predictions_dalc = dalc.predict(test.X)
            # Calculating Risk for Classic DALC
            risk_dalc = dalc.calc_risk(test.Y, predictions=predictions_dalc)

            # Predictions for Active DALC
            predictions = classifier.predict(test.X)
            # Calculating Risk for Active DALC
            risk = classifier.calc_risk(test.Y, predictions=predictions)

            # print('ROTATION({}) >> Test risk = '.format(i) + str(risk))
            # print('--------------------------------------------------')

            # Saving Results
            result.append(risk)
            classic_result.append(risk_dalc)

        # Calculating average
        result = np.asarray(result)
        average = np.average(result)
        classic_result = np.asarray(classic_result)
        classic_average = np.average(classic_result)
        text_file = open("results\\cost_results_custom.txt", "a")
        text_file.write("===================================================\n")
        text_file.write("MOON DATASET (ROTATION = {}) (COST ={})\n".format(70, i))
        text_file.write("ACTIVE DALC\n")
        text_file.write("Classification Risk = {}\n".format(str(average)))
        # text_file.write("---------------------------------------------------\n")
        # text_file.write("CLASSIC DALC\n")
        # text_file.write("Classification Risk = {}\n".format(str(risk_dalc)))
        text_file.write("===================================================\n")
        text_file.close()

        text_file = open("results\\cost_results_custom_numbers_only.txt", "a")
        text_file.write(str(average) + "\n")
        text_file.close()

        text_file = open("results\\classic.txt", "a")
        text_file.write(str(classic_average) + "\n")
        text_file.close()

        print("Classic DALC : {}".format(str(classic_average)))
        print("Active DALC : {}".format(str(average)))


def multiple_iterations_experiment(start_iter, end_iter):
    for i in range(start_iter, end_iter):
        result = list()
        classic_result = list()
        for j in range(0, 11):
            datasets = setup.generate_moon_dataset_save_all(200, 200, 1000, 70, 50, j)
            # Reading Datasets (Source, Target, Test)
            source, target, test = setup.read_all_data(50, j)

            # Executing Algorithm
            dalc, classifier, data, labels = active_dalc(source, target, test, 50, i
                                                         , 1.0, 1.0, 0.5
                                                         , str(70))
            # save_data(data, labels, str(i))

            # Predictions for Classic DALC
            predictions_dalc = dalc.predict(test.X)
            # Calculating Risk for Classic DALC
            risk_dalc = dalc.calc_risk(test.Y, predictions=predictions_dalc)

            # Predictions for Active DALC
            predictions = classifier.predict(test.X)
            # Calculating Risk for Active DALC
            risk = classifier.calc_risk(test.Y, predictions=predictions)

            # print('ROTATION({}) >> Test risk = '.format(i) + str(risk))
            # print('--------------------------------------------------')

            # Saving Results
            result.append(risk)
            classic_result.append(risk_dalc)

        # Calculating average
        result = np.asarray(result)
        average = np.average(result)
        classic_result = np.asarray(classic_result)
        classic_average = np.average(classic_result)
        text_file = open("results\\iterations_results.txt", "a")
        text_file.write("===================================================\n")
        text_file.write("MOON DATASET (ROTATION = {}) (COST ={})\n".format(70, 50))
        text_file.write("ACTIVE DALC\n")
        text_file.write("Classification Risk = {}\n".format(str(average)))
        # text_file.write("---------------------------------------------------\n")
        # text_file.write("CLASSIC DALC\n")
        # text_file.write("Classification Risk = {}\n".format(str(risk_dalc)))
        text_file.write("===================================================\n")
        text_file.close()

        text_file = open("results\\iterations_results_numbers_only.txt", "a")
        text_file.write(str(average) + "\n")
        text_file.close()

        text_file = open("results\\classic.txt", "a")
        text_file.write(str(classic_average) + "\n")
        text_file.close()

        print("Classic DALC : {}".format(str(classic_average)))
        print("Active DALC : {}".format(str(average)))


def run_active_dalc():
    # Reading Datasets (Source, Target, Test)
    source, target, test = setup.read_data()
    # setup.plot_datasets([(source.X, source.Y), (target.X, target.Y), (test.X, test.Y)], 'tmp')
    # Executing Algorithm
    cost = 50
    iterations = 5
    dalc, classifier, data, labels = active_dalc(source, target, test, cost, iterations
                                                 , 1.0, 1.0, 0.5
                                                 , 'manual_experiment')

    save_data(data, labels)
    # save_model(classifier, cost, iterations)

    # Predictions for Classic DALC
    predictions_dalc = dalc.predict(test.X)
    # Calculating Risk for Classic DALC
    risk_dalc = dalc.calc_risk(test.Y, predictions=predictions_dalc)
    print('Classic Test risk = ' + str(risk_dalc))
    # Predictions for Active DALC
    predictions = classifier.predict(test.X)
    # Calculating Risk for Active DALC
    risk = classifier.calc_risk(test.Y, predictions=predictions)

    # Calculating Risk
    print('Active Test risk = ' + str(risk))

    # plot_model(source.X, source.Y, classifier, test.X, 'test')


def custom_experiment(angle):
    # for i in range(10, 201, 10):
    #     print(i)
        i = 50
        result = list()
        classic_result = list()
        for j in range(0, 11):
            datasets = setup.generate_moon_dataset_save_all(200, 200, 1000, angle, i, j)
            # Reading Datasets (Source, Target, Test)
            source, target, test = setup.read_all_data(i, j)

            # Executing Algorithm
            dalc, classifier, data, labels = active_dalc(source, target, test, i, 5
                                                         , 1.0, 1.0, 0.5
                                                         , str(angle))
            # save_data(data, labels, str(i))

            # Predictions for Classic DALC
            predictions_dalc = dalc.predict(test.X)
            # Calculating Risk for Classic DALC
            risk_dalc = dalc.calc_risk(test.Y, predictions=predictions_dalc)

            # Predictions for Active DALC
            predictions = classifier.predict(test.X)
            # Calculating Risk for Active DALC
            risk = classifier.calc_risk(test.Y, predictions=predictions)

            # print('ROTATION({}) >> Test risk = '.format(i) + str(risk))
            # print('--------------------------------------------------')

            #Saving Results
            result.append(risk)
            classic_result.append(risk_dalc)

        # Calculating average
        result = np.asarray(result)
        average = np.average(result)
        classic_result = np.asarray(classic_result)
        classic_average = np.average(classic_result)
        text_file = open("results\\cost_results_custom.txt", "a")
        text_file.write("===================================================\n")
        text_file.write("MOON DATASET (ROTATION = {}) (COST ={})\n".format(angle, i))
        text_file.write("ACTIVE DALC\n")
        text_file.write("Classification Risk = {}\n".format(str(average)))
        # text_file.write("---------------------------------------------------\n")
        # text_file.write("CLASSIC DALC\n")
        # text_file.write("Classification Risk = {}\n".format(str(risk_dalc)))
        text_file.write("===================================================\n")
        text_file.close()

        text_file = open("results\\cost_results_custom_numbers_only.txt", "a")
        text_file.write(str(average)+"\n")
        text_file.close()

        text_file = open("results\\classic.txt", "a")
        text_file.write(str(classic_average) + "\n")
        text_file.close()

        print("Classic DALC : {}".format(str(classic_average)))
        print("Active DALC : {}".format(str(average)))


def main():
    datasets = setup.generate_moon_dataset(200, 200, 1000, 80)
    run_active_dalc()
    # custom_experiment(90)
    # for i in range(0, 21):
    # datasets = setup.generate_moon_dataset(200, 200, 1000, 50)
    # datasets2 = setup.generate_moon_dataset(200, 200, 1000, 50)
    #
    # setup.plot_datasets(datasets, '1')
    # setup.plot_datasets(datasets2, '2')
    # multiple_rotations_experiment(50, 5, 30, 100)
    # multiple_cost_experiment(190, 210)
    # multiple_iterations_experiment(1, 11)
    print("---------------------------------------------------")
    print("                    FINISHED")
    print("---------------------------------------------------")


if __name__ == "__main__":
    main()