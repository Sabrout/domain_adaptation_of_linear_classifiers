from sklearn import datasets
from math import sin, cos, radians
import dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from sklearn import svm


def rotate_point(point, angle, center_point=(0, 0)):
    """Rotates a point around center_point(origin by default)
    Angle is in degrees.
    Rotation is counter-clockwise
    """
    angle_rad = radians(angle % 360)
    # Shift the point so that center_point becomes the origin
    new_point = (point[0] - center_point[0], point[1] - center_point[1])
    new_point = (new_point[0] * cos(angle_rad) - new_point[1] * sin(angle_rad),
                 new_point[0] * sin(angle_rad) + new_point[1] * cos(angle_rad))
    # Reverse the shifting we have done
    new_point = (new_point[0] + center_point[0], new_point[1] + center_point[1])
    return new_point


def shift_point(point):
    new_point = (point[0] - 0.5, point[1] - 0.25)
    return new_point


def shift_dataset(dataset):
    for i in range(0, len(dataset[0])):
        dataset[0][i] = shift_point((dataset[0][i][0], dataset[0][i][1]))


def rotate_dataset(dataset, angle):
    for i in range(0, len(dataset[0])):
        dataset[0][i] = rotate_point((dataset[0][i][0], dataset[0][i][1]), angle)


def adapt_labels(dataset):
    for i in range(0, len(dataset[1])):
        if dataset[1][i] == 0:
            dataset[1][i] = -1


def clear_folder(path):
    files = glob.glob(path)
    for f in files:
        os.remove(f)


def clean_tmp():
    clear_folder("models/*")
    clear_folder("predictions/*")
    clear_folder("results/*")
    clear_folder("data/*")


def log_scale(start, end, step=0.1):
    scale = list()
    log_start = np.log10(start)
    log_end = np.log10(end)
    range = np.arange(log_start, log_end, step)
    for i in range:
        tmp = 10**i
        scale.append(tmp)
    return scale


def generate_moon_dataset(source_size=200, target_size=200, test_size=1000, rotation_angle=30):
    source_dataset = datasets.make_moons(n_samples=source_size, shuffle=True, noise=0.0, random_state=None)
    adapt_labels(source_dataset)
    shift_dataset(source_dataset)
    datasets.dump_svmlight_file(source_dataset[0], source_dataset[1], 'data\\source.svmlight', zero_based=True
                                , comment=None, query_id=None, multilabel=False)

    target_dataset = datasets.make_moons(n_samples=target_size, shuffle=True, noise=0.0, random_state=None)
    adapt_labels(target_dataset)
    shift_dataset(target_dataset)
    rotate_dataset(target_dataset, rotation_angle)
    datasets.dump_svmlight_file(target_dataset[0], target_dataset[1], 'data\\target.svmlight', zero_based=True
                                , comment=None, query_id=None, multilabel=False)

    test_dataset = datasets.make_moons(n_samples=test_size, shuffle=True, noise=0.0, random_state=1)
    adapt_labels(test_dataset)
    shift_dataset(test_dataset)
    rotate_dataset(test_dataset, rotation_angle)
    datasets.dump_svmlight_file(test_dataset[0], test_dataset[1], 'data\\test.svmlight', zero_based=True, comment=None,
                                query_id=None, multilabel=False)
    return source_dataset, target_dataset, test_dataset


def generate_moon_dataset_save_all(source_size=200, target_size=200, test_size=1000, rotation_angle=30, cost=0, run=0):
    source_dataset = datasets.make_moons(n_samples=source_size, shuffle=True, noise=0.02, random_state=None)
    adapt_labels(source_dataset)
    shift_dataset(source_dataset)
    datasets.dump_svmlight_file(source_dataset[0], source_dataset[1]
                                , 'data\custom\\source-cost{}-run{}.svmlight'.format(cost, run), zero_based=True
                                , comment=None, query_id=None, multilabel=False)

    target_dataset = datasets.make_moons(n_samples=target_size, shuffle=True, noise=0.02, random_state=2)
    adapt_labels(target_dataset)
    shift_dataset(target_dataset)
    rotate_dataset(target_dataset, rotation_angle)
    datasets.dump_svmlight_file(target_dataset[0], target_dataset[1],
                                'data\custom\\target-cost{}-run{}.svmlight'.format(cost, run), zero_based=True
                                , comment=None, query_id=None, multilabel=False)

    test_dataset = datasets.make_moons(n_samples=test_size, shuffle=True, noise=0.02, random_state=1)
    adapt_labels(test_dataset)
    shift_dataset(test_dataset)
    rotate_dataset(test_dataset, rotation_angle)
    datasets.dump_svmlight_file(test_dataset[0], test_dataset[1],
                                'data\custom\\test-cost{}-run{}.svmlight'.format(cost, run), zero_based=True, comment=None,
                                query_id=None, multilabel=False)
    return source_dataset, target_dataset, test_dataset


def plot_datasets(datasets, fig_name):
    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

    plt.subplot(221, aspect='equal')
    plt.title("Source Dataset", fontsize='small')
    X1, Y1 = datasets[0]
    plt.axis([-2.0, 2.0, -1.0, 1.0])
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
                s=25, edgecolor='k')

    plt.subplot(222, aspect='equal')
    plt.title("Target Dataset", fontsize='small')
    X2, Y2 = datasets[1]
    plt.axis([-1.5, 1.5, -1.25, 1.25])
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2,
                s=25, edgecolor='k')

    plt.subplot(223, aspect='equal')
    plt.title("Test Dataset",
              fontsize='small')
    X3, Y3 = datasets[2]
    plt.axis([-1.5, 1.5, -1.25, 1.25])
    plt.scatter(X3[:, 0], X3[:, 1], marker='o', c=Y3,
                s=25, edgecolor='k')

    plt.savefig('results/dataset_plots/'+ fig_name + 'datasets_display.png')
    plt.show()
    # plt.close(fig)


def plot_amazon(datasets):
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

    plt.subplot(121, aspect='equal')
    plt.title("Source Dataset", fontsize='small')
    X1 = datasets[0].X
    Y1 = datasets[0].Y
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
                s=25, edgecolor='k')

    plt.subplot(122, aspect='equal')
    plt.title("Target Dataset", fontsize='small')
    X1 = datasets[1].X
    Y1 = datasets[1].Y
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
                s=25, edgecolor='k')

    plt.savefig('results/amazon_display.png')
    plt.show()


def dalc_tune(b_min, b_max, c_min, c_max, b_step=0.1, c_step=0.1):
    # B-VALUE & C-VALUE TUNING
    b_range = log_scale(b_min, b_max, b_step)
    c_range = log_scale(c_min, c_max, c_step)
    g_range = log_scale(0.5, 1.5, 0.1)
    for i in b_range:
        for j in c_range:
            for k in g_range:
                # Writing Template
                template = ""
                template += "---------------------------------------------------\n"
                template += "MOONS DATABASE\n"
                template += "CASE : B = {}, C = {}, Gamma = {}\n".format(i, j, k)
                template += "---------------------------------------------------\n"
                text_file = open("results\\Classification.txt", "a")
                text_file.write(template)
                text_file.close()
                text_file = open("results\\Validation.txt", "a")
                text_file.write(template)
                text_file.close()
                # Training the model
                os.system("python dalc_learn.py -f svmlight -b {} -c {} -k rbf -g {} -m models\model-b{}c{}g{}.bin "
                          "data\source.svmlight data\\target.svmlight".format(i, j, k, i, j, k))
                # Testing the model
                command = ""
                command += "python dalc_classify.py -f svmlight -m models\model-b{}c{}g{}.bin " \
                           "-p predictions\pred-b{}c{}g{}.bin data\\test.svmlight >> ".format(i, j, k, i, j, k)
                command += "results\\Classification.txt"
                os.system(command)
                # Validating the model
                command = ""
                command += "python dalc_reverse_cv.py -b {} -c {} -k rbf -g {} -f svmlight " \
                           "data\source.svmlight data\\target.svmlight >> results\Validation.txt".format(i, j, k)
                os.system(command)


def dalc_amazon(b_min, b_max, c_min, c_max, b_step=1.0, c_step=1.0):
    # B-VALUE & C-VALUE TUNING
    b_range = np.arange(b_min, b_max, b_step)
    c_range = np.arange(c_min, c_max, c_step)
    g_range = np.arange(0.1, 1.0, 0.5)
    for i in b_range:
        for j in c_range:
            for k in g_range:
                text_file = open("results\\amazon.txt", "a")
                text_file.write("---------------------------------------------------\n")
                text_file.write("AMAZON DATABASE\n")
                text_file.write("CASE : B = {}, C = {}, Gamma = {}\n".format(i, j, k))
                text_file.write("---------------------------------------------------\n")
                text_file.close()
                # Training the model
                os.system("python dalc_learn.py -f svmlight -b {} -c {} -k rbf -g {} -m models\\amazon-b{}c{}g{}.bin "
                          "amazon\source.svmlight amazon\\target.svmlight".format(i, j, k, i, j, k))
                # Testing the model
                command = ""
                command += "python dalc_classify.py -f svmlight -m models\\amazon-b{}c{}g{}.bin " \
                           "-p predictions\\amazon-b{}c{}g{}.bin amazon\\test.svmlight >> ".format(i, j, k, i, j, k)
                command += "results\\amazon.txt"
                os.system(command)


def read_data():
    source = dataset.dataset_from_svmlight_file('data\source.svmlight', 2)
    target = dataset.dataset_from_svmlight_file('data\\target.svmlight', 2)
    test = dataset.dataset_from_svmlight_file('data\\test.svmlight', 2)
    return source, target, test


def read_all_data(cost=0, run=0):
    source = dataset.dataset_from_svmlight_file('data\custom\source-cost{}-run{}.svmlight'.format(cost, run), 2)
    target = dataset.dataset_from_svmlight_file('data\custom\\target-cost{}-run{}.svmlight'.format(cost, run), 2)
    test = dataset.dataset_from_svmlight_file('data\custom\\test-cost{}-run{}.svmlight'.format(cost, run), 2)
    return source, target, test


def read_amazon():
    source = dataset.dataset_from_svmlight_file('amazon\source.svmlight')
    target = dataset.dataset_from_svmlight_file('amazon\\target.svmlight')
    return source, target


def extract_model():
    # READING VALIDATION FILE
    validation = open("results\\Validation.txt", "r")
    lines = validation.readlines()
    # Matrix of Models such as (B, C, Gamma, Validation Score, Classification Score, Standard Deviation)
    models = list()
    for i in range(2, len(lines), 145):
        result = re.findall(r"\d+\.\d*", lines[i])
        if len(result) == 3:
            risk = re.findall(r"\d+\.\d*", lines[i+142])
            result.append(risk[0])
            risk = re.findall(r"\d+\.\d*", lines[i+140])
            result.append(risk[0])
            models.append(result)
        else:
            print('VALIDATION - LINE READING ERROR')
    validation.close()

    # READING CLASSIFICATION FILE
    classification = open("results\\Classification.txt", "r")
    lines = classification.readlines()
    j = 0
    for i in range(12, len(lines), 13):
        result = re.findall(r"\d+\.\d*", lines[i])
        if len(result) == 1:
            models[j].append(result[0])
            j += 1
        else:
            print('CLASSIFICATION - LINE READING ERROR')
    classification.close()

    # EXTRACTING BEST MODELS
    matrix = np.asarray(models, dtype=np.float32)
    sorted = np.lexsort((matrix[:, 4], matrix[:, 5], matrix[:, 3]))
    best_model = matrix[sorted][0]
    print("---------------------------------------------------")
    print("OPTIMAL MODEL FOR MOON DATASET")
    print("CASE : B = {}, C = {}, Gamma = {}\n".format(best_model[0], best_model[1], best_model[2]))
    print("Validation Risk = {}".format(best_model[3]))
    print("Standard Deviation = {}".format(best_model[5]))
    print("Classification Risk = {}".format(best_model[4]))
    print("---------------------------------------------------")
    return best_model


def plot_model(dataset, dataset2, model):
    # We will now create a 2d graphic to illustrate the learning result
    # We create a mesh to plot in
    h = .02  # grid step
    x_min = dataset[0][:, 0].min() - 1
    x_max = dataset[0][:, 0].max() + 1
    y_min = dataset[0][:, 1].min() - 1
    y_max = dataset[0][:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # The grid is created, the intersections are in xx and yy

    Z2d = model.predict(np.c_[xx.ravel(), yy.ravel()])  # We predict all the grid
    Z2d = Z2d.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z2d, cmap=plt.cm.Paired)
    # We plot also the training points
    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=dataset[1], cmap=plt.cm.coolwarm)
    plt.scatter(dataset2[0][:, 0], dataset2[0][:, 1], c=dataset2[1], cmap=plt.cm.coolwarm)

    plt.show()
    # h_sep = svm.SVC(kernel='linear', C=1.0)
    # h_sep.fit(datasets[0][0], datasets[0][1])
    # plot_model(datasets[0], datasets[1], h_sep)


def main():
    # clean_tmp()
    datasets = generate_moon_dataset(200, 200, 1000, 80)
    # plot_datasets(datasets, 'rotation0')
    # dalc_tune(0.5, 1.0, 0.5, 1.0, 0.5, 0.5)
    # models = extract_model()

    # amazon = read_amazon()
    # plot_amazon(amazon)
    # dalc_amazon(1, 5, 0.1, 1.5, 2, 0.5)

    # result = log_scale(0.1, 1.0, 0.05)
    # for i in result:
    #     print(i)

    print("---------------------------------------------------")
    print("                    FINISHED")
    print("---------------------------------------------------")


if __name__ == "__main__":
    main()
