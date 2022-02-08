'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import numpy as np
import matplotlib.pyplot as plt
import submission
import helper
import util


pts = np.load('../data/some_corresp.npz')
pts1 = pts["pts1"]
pts2 = pts["pts2"]

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
M = np.max(np.shape(im1))

F = submission.eightpoint(pts1, pts2, M)

intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics["K1"]
K2 = intrinsics["K2"]
E = submission.essentialMatrix(F, K1, K2)

M1 = np.eye(4)[0:3, :]
C1 = K1 @ M1

M2all = helper.camera2(E)
num = np.size(M2all, axis = 2)

for i in range(num):
    M2 = M2all[:,:,i]
    C2 = K2 @ M2
    w, err = submission.triangulate(C1, pts1, C2, pts2)
    # print("Error = " ,err)
    zmin = np.min(w[:,-1])
    # print("zmin = ", zmin)
    if(zmin > 0):
        break;
        
# np.savez('q3_3.npz', M2 = M2, C2 = C2, P = w)
