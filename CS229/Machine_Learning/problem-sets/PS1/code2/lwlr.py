"""
Created on Sat Dec 11 17:58:10 2021
@author: leexuewei
"""
import numpy as np

# define h(theta, X)
import numpy as np

#define h(theta, X)
def h(theta, X):
    return 1 / (1 + np.exp(- X.dot(theta)))

tau = 1
Lambda = 0.0001
threshold = 1e-6

def lwlr(X_train, y_train, x, tau):
    m, d = X_train.shape
    #initialize
    theta = np.zeros(d)
    #compute weights
    norm = np.sum((X_train - x) ** 2, axis=1)
    W = np.exp(- norm / (2 * tau ** 2))
    #initialize grad 
    g = np.ones(d)
    
    while np.linalg.norm(g) > threshold:
        #compute h(theta, X)
        h_X = h(theta, X_train)
        #grad
        z = W * (y_train - h_X)
        g = X_train.T.dot(z) - Lambda * theta
        #Hessian Matrix
        D = - np.diag(W * h_X * (1 - h_X))
        H = X_train.T.dot(D).dot(X_train) - Lambda * np.eye(d)
        
        #update
        theta -= np.linalg.inv(H).dot(g)
    
    ans = (theta.dot(x) > 0).astype(np.float64)
    return ans
