import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from lwlr import lwlr
from plot_lwlr import plot_lwlr

Lambda = 0.0001
threshold = 1e-6

Tau = [0.01, 0.05, 0.1, 0.5, 1, 5]
X, y = load_data()
for tau in Tau:
    plot_lwlr(X, y, tau)