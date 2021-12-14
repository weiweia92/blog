"""
Created on Sat Dec 11 17:58:10 2021
@author: leexuewei
"""
import numpy as np

def load_data():
    X = np.loadtxt('data/x.dat')
    y = np.loadtxt('data/y.dat')
    
    return X, y