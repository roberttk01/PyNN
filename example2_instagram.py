from PyNN import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# X = (time posted, impressions, reach), y = Number of likes
X = np.array(([20.92, 40, 36], [21.65, 74, 67], [9.92, 107, 98], [10.7, 205, 171], [15.5, 37, 30], [20.53, 70, 63], [20.8, 89, 64], [11.6, 79, 59],
              [19.02, 75, 54], [10.93, 55, 42], [16.63, 63, 49], [11.00, 55, 46]), dtype=float)
y = np.array(([28], [28], [31], [20], [30], [34], [34], [41], [47], [40], [68], [23]), dtype=float)

# Normalize
X = X/np.amax(X, axis=0)
y = y/np.amax(y, axis=0) # Max test score is 100
hl_num = 4 # Default should be len(X[0] + 1)


# Instantiate Neural Network and Train with data
NN = NeuralNetwork(len(X[0]), len(y[0]), hl_num)
T = trainer(NN)
T.train(X, y)



size = 1500

post_time = np.linspace(8, 21, size)
engagement = np.linspace(1, 250, size)
reach = np.linspace(1, 200, size)

post_time_norm = post_time / 21
engagement_norm = engagement / 250
reach_norm = reach / 200

a, b, c = np.meshgrid(post_time_norm, engagement_norm, reach_norm)
allInputs = np.zeros((a.size, 3))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()
allInputs[:, 2] = c.ravel()

predicted_likes = NN.forward(allInputs)

yy = np.dot(engagement_norm.reshape(size, 1), np.ones((1, size)))
xx = np.dot(post_time_norm.reshape(size, 1), np.ones((1, size))).T

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx*21, yy*250, predicted_likes.reshape(size, size)*70, cmap=cm.jet)
ax.set_xlabel('Time Posted')
ax.set_ylabel('Total Engagement')
ax.set_zlabel('Projected Likes')

plt.show()

