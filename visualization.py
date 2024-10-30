import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("data.txt", 'rb') as file:
    data = pickle.load(file)

path = data['path']
ref = data['ref']

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(ref[:,0], ref[:,1], label="reference")
ax.plot(path[:,1], path[:,2], label="car")
# ax.axis('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.legend()
plt.show()