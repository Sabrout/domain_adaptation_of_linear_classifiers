import numpy as np
from dataset import Dataset
from dalc import Dalc


def load_dat(path):
    content = np.loadtxt(path, dtype=int)
    if content.ndim == 1:
        dataset = np.empty((len(content), ), dtype=int)
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


X1 = load_dat('data/books/X.dat')
Y1 = load_dat('data/books/Y.dat')

source_dataset = Dataset(X1[:len(Y1)], Y1)

X2 = load_dat('data/dvd/X.dat')

target_dataset = Dataset(X2[:len(Y1)])

print((len(X1), len(Y1), len(X2)))
print((len(source_dataset.X), len(source_dataset.Y), len(target_dataset.X)))

# print(source_dataset.get_nb_examples())
# print(source_dataset.get_nb_features())
# print(source_dataset.select_examples([2]))

dalc = Dalc()
result = dalc.learn(source_dataset, target_dataset)
print(result)