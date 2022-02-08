'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import matplotlib.pyplot as plt
import numpy as np
import submission
import helper
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from plotly.offline import plot

pts = np.load('../data/some_corresp.npz')
pts1 = pts["pts1"]
pts2 = pts["pts2"]

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
M = np.max(np.shape(im1))

F = submission.eightpoint(pts1, pts2, M)
print("F =", F)

intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics["K1"]
K2 = intrinsics["K2"]
E = submission.essentialMatrix(F, K1, K2)

print("E = ", E)

templeCoords = np.load('../data/templeCoords.npz')
x1 = templeCoords["x1"]
y1 = templeCoords["y1"]

x2 = np.zeros(np.size(x1))
y2 = np.zeros(np.size(y1))


for i in range(np.size(x1)):
    x2[i], y2[i] = submission.epipolarCorrespondence(im1, im2, F, x1[i][0], y1[i][0])
                
x2 = x2.reshape(np.size(x2), 1)
y2 = y2.reshape(np.size(y2), 1)

M1 = np.eye(4)[0:3, :]
C1 = K1 @ M1

M2all = helper.camera2(E)
num = np.size(M2all, axis = 2)

pts1 = np.hstack((x1,y1))
pts2 = np.hstack((x2,y2))

for i in range(num):
    M2 = M2all[:,:,i]
    C2 = K2 @ M2
    w, err = submission.triangulate(C1, pts1, C2, pts2)
    # print("Error = " ,err)
    zmin = np.min(w[:,-1])
    # print("zmin = ", zmin)
    if(zmin > 0):
        break;
        
# np.savez('q4_2.npz', F = F, M1 = M1, M2 = M2, C1 = C1, C2 = C2)
        
#
# figure = px.scatter_3d(x = w[:,0], y = w[:,1], z = w[:,2])
# figure.update_traces(marker =dict(size=2))
# figure.update_layout(scene = dict(zaxis = dict(range =[3,4.2])))

# plot(figure)


# figure = px.scatter_3d(x = w[:,0], y = w[:,1], z = -w[:,2])
# figure.update_traces(marker =dict(size=3))
# figure.update_layout(scene = dict(zaxis = dict(range =[-4.2,-3.4])))

# plot(figure)




