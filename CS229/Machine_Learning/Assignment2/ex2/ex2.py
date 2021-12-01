import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from plotData import *
import costFunction as cf
import plotDecisionBoundary as pdb
import predict as predict
from sigmoid import *

plt.ion()
# Load data
# The first two columns contain the exam scores and the third column contains the label.
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

# ===================== Part 1: Plotting =====================
print('Plotting Data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plot_data(X, y)
plt.axis([30, 100, 30, 100])
plt.legend(['Admitted', 'Not admitted'], loc=1)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Compute Cost and Gradient =====================
# In this part of the exercise, you will implement the cost and gradient
# for logistic regression. You need to complete the code in
# costFunction.py

# Setup the data array appropriately, and add ones for the intercept term
(m, n) = X.shape

# Add intercept term
X = np.c_[np.ones(m), X]

# Initialize fitting parameters
initial_theta = np.zeros(n + 1)

# Compute and display initial cost and gradient
cost, grad = cf.cost_function(initial_theta, X, y)

np.set_printoptions(formatter={'float': '{: 0.4f}\n'.format})

print('Cost at initial theta (zeros): {:0.3f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): \n{}'.format(grad))
print('Expected gradients (approx): \n-0.1000\n-12.0092\n-11.2628')

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Optimizing using fmin_bfgs =====================
# In this exercise, you will use a built-in function (opt.fmin_bfgs) to find the
# optimal parameters theta
def cost_func(theta):
    return cf.cost_function(theta, X, y)[0]

def grad_func(theta):
    return cf.cost_function(theta, X, y)[1]

# Run fmin_bfgs to obtain the optimal theta
theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func, x0=initial_theta, maxiter=400, full_output=True, disp=False)

print('Cost at theta found by fmin: {:0.4f}'.format(cost))
print('Expected cost (approx): 0.203')
print('theta: \n{}'.format(theta))
print('Expected Theta (approx): \n-25.161\n0.206\n0.201')

# Plot boundary
pdb.plot_decision_boundary(theta, X, y)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
input('Program paused. Press ENTER to continue')