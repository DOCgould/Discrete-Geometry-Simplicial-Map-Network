from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# vertices of a pyramid
v = np.array([[0,0,0], #0
              [2,0,0], #1
              [0,2,0], #2
              [2,2,0], # 11
              [1,1,1]])#3

ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])

# generate list of sides' polygons of our pyramid
verts = [
         [v[0],v[1],v[2],v[3]], # Base
         [v[0],v[2],v[4]],
         [v[0],v[1],v[4]],
         [v[3],v[2],v[4]],
         [v[3],v[1],v[4]]
        ]

# -- Toal is Five -- #


# plot sides
ax.add_collection3d(Poly3DCollection(verts, 
 facecolors='black', linewidths=1, edgecolors='k', alpha=.25))

plt.show()
