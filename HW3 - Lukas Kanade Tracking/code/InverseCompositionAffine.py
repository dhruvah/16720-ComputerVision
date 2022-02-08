import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array]
    """

    # put your implementation here
    M = np.eye(3)
    
    p = M[0:2, 0:3].flatten()
    
    x1, y1, x2, y2 = 0, 0, It.shape[1] -1, It.shape[0] -1
    
    It_spline=RectBivariateSpline(np.arange(It.shape[0]),
                                  np.arange(It.shape[1]), 
                                  It)
    
    It1_spline=RectBivariateSpline(np.arange(It1.shape[0]),
                                   np.arange(It1.shape[1]),
                                   It1)
    
    dp = np.array([float("inf")]* 6)
    
    count = 0
    
    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    X, Y = np.meshgrid(x,y)
    
    dwx = It_spline.ev(Y,X,dx = 0, dy = 1).flatten()
    dwy = It_spline.ev(Y,X,dx = 1, dy = 0).flatten()
    
    A = np.array([dwx*X.flatten(),
                  dwx*Y.flatten(),
                  dwx,
                  dwy*X.flatten(),
                  dwy*Y.flatten(),
                  dwy]).T
    
    
    while np.linalg.norm(dp)>= threshold and count <= num_iters:
        
        count += 1
        
        warpX = p[0] * X + p[1] * Y + p[2]
        warpY = p[3] * X + p[4] * Y + p[5]
        
        commonpts = np.where((warpX >= x1) & (warpX <= x2) & 
                             (warpY >= y1) & (warpY <= y2), True, False) 
        
        warpX, warpY = warpX[commonpts], warpY[commonpts]
        It1x_warp = It1_spline.ev(warpY, warpX)  
        b = (It1x_warp - It[commonpts]).flatten()
    
        
        # print(np.shape(b))
        A_c = A[commonpts.flatten()]
        dp = (np.linalg.inv(A_c.T @ A_c)) @ (A_c.T @ b)
        # print(np.shape(dp))
        
        M = np.vstack((p.reshape(2,3), M[2,0:3]))
        dM = np.vstack((dp.reshape(2,3), M[2,0:3]))
        dM[0,0] += 1
        dM[1,1] += 1
        # print(np.shape(dM))
        
        M = M @ np.linalg.inv(dM)
        
        p = M[0:2, 0:3].flatten()

    return M
