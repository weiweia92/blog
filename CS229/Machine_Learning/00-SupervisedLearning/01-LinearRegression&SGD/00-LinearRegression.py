"""
  author: Lee Xuewei
  since: 2021-11-17 20:16:17
  description: linear regression(done)
"""
from scipy.sparse import data
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time
import ipdb

# make dummy(虚拟) regression data
X, y = datasets.make_regression(n_samples=250, n_features=1, noise=20, random_state=0, bias=50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #X_train:(175,1)  y_train:(175) 
# for theata0, x0 = 1
o_train = np.ones([X_train.shape[0], 1], dtype=X_train.dtype)
o_test = np.ones([X_test.shape[0], 1], dtype=X_test.dtype)
X_train = np.concatenate((o_train, X_train), axis=1)  #X_train.shape:(175,2) two features include x0
X_test = np.concatenate((o_test, X_test), axis=1)

# show data
plt.scatter(X_train[:,1], y_train, c='orange', edgecolors='white')
plt.scatter(X_test[:,1], y_test, c='red', edgecolors='white')
plt.xlim((-3, 3))
plt.ylim((-25, 125))
plt.show()  # pic/Figure_1.png

# learnable parameter theta
THETA = np.zeros([2, 1], dtype=np.float32)
# learning rate
lr_sto = 0.01
lr_bat = 0.0001
# Epoch
Epoch = 2

def hypothesis(x, θ):
    h_x = np.matmul(θ.T, x)
    return h_x

#single training example
def loss_function(h_x, y):
    J = 1/2 * (h_x - y) * (h_x - y)
    return J