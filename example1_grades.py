from PyNN import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3, 5], [5, 1], [10, 2], [8, 6], [12, 6]), dtype=float)
y = np.array(([75], [82], [93], [99], [97]), dtype=float)

# Normalize
X = X/np.amax(X, axis=0)
y = y/100 # Max test score is 100
hl_num = 10


# Instantiate Neural Network and Train with data
NN = NeuralNetwork(len(X[0]), len(y[0]), hl_num)
T = trainer(NN)
T.train(X, y)

size = 1000

hours_slept = np.linspace(0, 24, size)
hours_studied = np.linspace(0, 10, size)
hours_slept = hours_slept / 24
hours_studied = hours_studied / 10

a, b = np.meshgrid(hours_slept, hours_studied)
time = np.zeros((a.size, 2))
time[:, 0] = a.ravel()
time[:, 1] = b.ravel()

grades = NN.forward(time)

yy = np.dot(hours_studied.reshape(size, 1), np.ones((1, size)))
xx = np.dot(hours_slept.reshape(size, 1), np.ones((1, size))).T

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx*24, yy*10, 100 * grades.reshape(size, size), cmap=cm.jet)
ax.set_xlabel('Hours Slept')
ax.set_ylabel('Hours Studied')
ax.set_zlabel('Grade')

plt.show()

