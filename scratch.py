import scipy.io as sio

content = sio.loadmat('data/books.dvd.X.mat')

for i in content.get('X'):
    print(i)