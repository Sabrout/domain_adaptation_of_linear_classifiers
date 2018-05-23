from sklearn import datasets
from math import sin, cos, radians
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


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


def rotate_dataset(dataset, angle):
    for i in range(0, len(dataset[0])):
        dataset[0][i] = rotate_point((dataset[0][i][0], dataset[0][i][1]), angle)


def generate_moon_dataset(source_size=200, target_size=200, test_size=1000):
    source_dataset = datasets.make_moons(n_samples=source_size, shuffle=None, noise=None, random_state=None)
    datasets.dump_svmlight_file(source_dataset[0], source_dataset[1], 'data\\source.svmlight', zero_based=True, comment=None,
                                query_id=None, multilabel=False)

    target_dataset = datasets.make_moons(n_samples=target_size, shuffle=None, noise=None, random_state=None)
    # rotate_dataset(target_dataset, 30)
    datasets.dump_svmlight_file(target_dataset[0], target_dataset[1], 'data\\target.svmlight', zero_based=True, comment=None,
                                query_id=None, multilabel=False)

    test_dataset = datasets.make_moons(n_samples=test_size, shuffle=None, noise=None, random_state=None)
    # rotate_dataset(test_dataset, 30)
    datasets.dump_svmlight_file(test_dataset[0], test_dataset[1], 'data\\test.svmlight', zero_based=True, comment=None,
                                query_id=None, multilabel=False)
    return source_dataset, target_dataset, test_dataset


def plot_datasets(datasets):
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

    plt.subplot(221)
    plt.title("Source Dataset", fontsize='small')
    X1, Y1 = datasets[0]
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
                s=25, edgecolor='k')

    plt.subplot(222)
    plt.title("Target Dataset", fontsize='small')
    X1, Y1 = datasets[1]
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
                s=25, edgecolor='k')

    plt.subplot(223)
    plt.title("Test Dataset",
              fontsize='small')
    X2, Y2 = datasets[2]
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2,
                s=25, edgecolor='k')

    plt.show()
    plt.savefig('datasets.png')


def dalc_tune(b_min, b_max, c_min, c_max, b_step=1, c_step=1):
    # B-VALUE & C-VALUE TUNING
    b_range = np.arange(b_min, b_max, b_step)
    c_range = np.arange(c_min, c_max, c_step)
    g_range = np.arange(1, 5)
    for i in b_range:
        for j in c_range:
            for k in g_range:
                text_file = open("results\\results.txt", "a")
                text_file.write("---------------------------------------------------\n")
                text_file.write("MOONS DATABASE\n")
                text_file.write("CASE : B = {}, C = {}, Gamma = {}\n".format(i, j, k))
                text_file.write("---------------------------------------------------\n")
                text_file.close()
                # Training the model
                os.system("python dalc_learn.py -f svmlight -b {} -c {} -k rbf -g {} -m models\model-b{}c{}g{}.bin "
                          "data\source.svmlight data\\target.svmlight".format(i, j, k, i, j, k))
                # Testing the model
                command = ""
                command += "python dalc_classify.py -f svmlight -m models\model-b{}c{}g{}.bin " \
                           "-p predictions\pred-b{}c{}g{}.bin data\\test.svmlight >> ".format(i, j, k, i, j, k)
                command += "results\\results.txt"
                os.system(command)


def clear_folder(path):
    files = glob.glob(path)
    for f in files:
        os.remove(f)


def clean_tmp():
    clear_folder("/models/*")
    clear_folder("/predictions/*")
    clear_folder("/results/*")
    clear_folder("/data/*")


def main():
    clean_tmp()
    datasets = generate_moon_dataset()
    plot_datasets(datasets)
    # dalc_tune(1, 3, 1, 3)
    print("---------------------------------------------------")
    print("                    FINISHED")
    print("---------------------------------------------------")


if __name__ == "__main__": main()