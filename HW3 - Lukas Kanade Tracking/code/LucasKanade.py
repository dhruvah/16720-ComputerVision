import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here
    p = p0
    
    x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
    
    It_spline=RectBivariateSpline(np.arange(It.shape[0]),
                                  np.arange(It.shape[1]), 
                                  It)
    
    It1_spline=RectBivariateSpline(np.arange(It1.shape[0]),
                                   np.arange(It1.shape[1]),
                                   It1)
    
    dp = np.array([float("inf"), float("inf")])
    
    count = 0
    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    x, y = np.meshgrid(x,y)
    
    while np.linalg.norm(dp)>= threshold and count <= num_iters:
        count += 1
        
        Itx = It_spline.ev(y, x).flatten()
        It1x_warp = It1_spline.ev(y+p[1], x+p[0]).flatten()
        A = np.array([
            It1_spline.ev(y+p[1], x+p[0], dx = 0, dy = 1).flatten(),
            It1_spline.ev(y+p[1], x+p[0], dx = 1, dy = 0).flatten()]).T
        
        # print(np.shape(A))
    
        b = (Itx - It1x_warp).flatten()
        
        # print(np.shape(b))
        
        dp = (np.linalg.inv(A.T @ A)) @ (A.T @ b)
        
        p+=dp
        
    
    return p
    
