import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
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
   
    
    while np.linalg.norm(dp)>= threshold and count <= num_iters:
        x = np.arange(x1, x2 + 1)
        y = np.arange(y1, y2 + 1)
        X, Y = np.meshgrid(x,y)
        count += 1
        
        warpX = p[0] * X + p[1] * Y + p[2]
        warpY = p[3] * X + p[4] * Y + p[5]
        
        commonpts = np.where((warpX >= x1) & (warpX <= x2) & 
                             (warpY >= y1) & (warpY <= y2), True, False) 
        
        X, Y = X[commonpts], Y[commonpts]
        warpX, warpY = warpX[commonpts], warpY[commonpts]
        
        dwx = It1_spline.ev(warpY, warpX, dx = 0, dy = 1).flatten()
        dwy = It1_spline.ev(warpY, warpX, dx = 1, dy = 0).flatten()
        
        It1x_warp = It1_spline.ev(warpY, warpX)
        
        A = np.array([dwx*X,
                      dwx*Y,
                      dwx,
                      dwy*X,
                      dwy*Y,
                      dwy]).T
        
        b = (It[commonpts] - It1x_warp).flatten()
        # print(np.shape(A))
    
        
        # print(np.shape(b))
        
        dp = (np.linalg.inv(A.T @ A)) @ (A.T @ b)
        # print(np.shape(dp))
        p+=dp
        
    M = np.vstack((p.reshape(2,3), M[2,0:3]))

    return M
