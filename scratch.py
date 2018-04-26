import numpy as np
from dataset import Dataset


def load_dat(path):
    content = np.loadtxt(path, dtype=int)
    if content.ndim == 1:
        dataset = np.empty((len(content), 1), dtype=int)
    else:
        dataset = np.empty(content.shape, dtype=int)
    for i in range(0, content.__len__()):
        if content.ndim == 1:
            dataset[i] = content[i]
        else:
            counter = 0
            for j in content[i]:
                if counter == 3:
                    break
                if j == ' ':
                    continue
                else:
                    dataset[i][counter] = j
                    counter += 1
    return dataset


X = load_dat('data/books/X.dat')
Y = load_dat('data/books/Y.dat')

dataset = Dataset(X, Y)

print(dataset.get_nb_examples())
print(dataset.get_nb_features())
print(dataset.select_examples([2]))