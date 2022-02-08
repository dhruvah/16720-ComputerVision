"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import helper
import cv2
import util
import scipy.ndimage as ndimage
from multiprocessing import Pool


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    x1 = pts1[:,0]/M
    y1 = pts1[:,1]/M
    x2 = pts2[:,0]/M
    y2 = pts2[:,1]/M
    
    U = np.vstack((x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, np.ones(np.shape(x1)))).T
    T = np.array([[1/M, 0, 0], 
                  [0, 1/M, 0],
                  [0, 0, 1]])
                  
    u, s, v = np.linalg.svd(U)
    F = v[-1, :].reshape(3,3)
    
    F = util.refineF(F, pts1/M, pts2/M)
    F = util._singularize(F)
    
    F = T.T @ F @ T
    
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    
    E = K2.T @ F @ K1
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    
    nPoints = np.size(x1)
    
    w = np.zeros((nPoints, 3))
    
    A1 = np.vstack((C1[2,0] * x1 - C1[0,0], C1[2,1] * x1 - C1[0,1], C1[2,2] * x1 - C1[0,2], C1[2,3] * x1 - C1[0,3])).T
    A2 = np.vstack((C1[2,0] * y1 - C1[1,0], C1[2,1] * y1 - C1[1,1], C1[2,2] * y1 - C1[1,2], C1[2,3] * y1 - C1[1,3])).T
    A3 = np.vstack((C2[2,0] * x2 - C2[0,0], C2[2,1] * x2 - C2[0,1], C2[2,2] * x2 - C2[0,2], C2[2,3] * x2 - C2[0,3])).T
    A4 = np.vstack((C2[2,0] * y2 - C2[1,0], C2[2,1] * y2 - C2[1,1], C2[2,2] * y2 - C2[1,2], C2[2,3] * y2 - C2[1,3])).T
    
    err = 0
    for i in range(nPoints):
        A = np.vstack((A1[i,:], A2[i,:], A3[i,:], A4[i,:]))
        u, s, v = np.linalg.svd(A)
        w[i,:] = v[-1,0:3] / v[-1,3:4]
        
        w_homo = np.hstack((w[i,:], 1)).reshape(4,1)
        
        # if i == 10:
        #     print("w = ", w[i,:])
        #     print("whomo = ", w_homo)
        x1i_hat = (C1 @ w_homo).reshape(3,1)
        x2i_hat = (C2 @ w_homo).reshape(3,1)
        
        x1i_hat = x1i_hat[0:2, :] / x1i_hat[-1, :]
        x2i_hat = x2i_hat[0:2, :] / x2i_hat[-1, :]
        
        points1 = (pts1[i]).reshape(2,1)
        points2 = (pts2[i]).reshape(2,1)
        
        err += np.sum(np.square(x1i_hat - points1) + np.square(x2i_hat - points2))
    
    return w, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):

    p = np.array([x1, y1, 1])
    line = F @ p
    H = im1.shape[0]
    W = im1.shape[1]
    
    liney = np.arange(H)
    linex = np.round(-(line[1]*liney + line[2])/line[0]).astype(np.int)
    
    im1 = ndimage.gaussian_filter(im1, sigma = 1)
    im2 = ndimage.gaussian_filter(im2, sigma = 1)
    
    size = 10
    patch1 = im1[y1 - size: y1 + size +1, x1 - size: x1 + size +1]
    
    minerror = 1000000
    index = 0
    for i in range(np.size(linex)):
        x2 = linex[i]
        y2 = liney[i]
        # print(x2, y2)
        if(x2>=size and x2 < W-size and y2>=size and y2<H-size):
            patch2 = im2[y2-size:y2+size+1, x2-size:x2+size+1] 
            diff = (patch1 - patch2).flatten()
            err = np.linalg.norm(diff, 2)
            if err < minerror:
                minerror = err
                index = i
    
    x2 = linex[index]
    y2 = liney[index]
    
    # print("x1, y1 = ", x1, y1)
    # print("x2, y2 = ", x2, y2)
    
    return x2, y2
    

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(args):
    pts1, pts2, M, nIters, tol = args
    # Replace pass by your implementation
    N1 = np.size(pts1[:,0])
    N2 = np.size(pts2[:,0])
    

    inliers = np.zeros(N1)
    maxnum = -1
    bestF = None
    
    x1_homo = np.hstack((pts1, np.ones(N1).reshape(N1,1)))
    x2_homo = np.hstack((pts2, np.ones(N2).reshape(N2,1)))
    
    index = np.random.choice(N2, 8, replace = False)
    for i in range(nIters):
        
        x1 = pts1[index]
        x2 = pts2[index]
        
        F = eightpoint(pts1, pts2, M)
        
        x1_pred_homo = F @ x2_homo.T
        scale = x1_pred_homo[-1,:]
        x1_pred_homo = x1_pred_homo/scale
        x1_pred_homo = x1_pred_homo.T
        diff = (x1_pred_homo - x1_homo)[:,[0,1]]
        normdiff = np.linalg.norm(diff, axis = 1)
        inlier = np.where(normdiff < tol , 1, 0)
        inliers_index = np.where(normdiff < tol)
        
        num_inliers = np.sum(inlier)
        
        if num_inliers > maxnum:
            maxnum = num_inliers
            bestF = F
            inliers = inlier
            
        if i > 8:
            index = np.random.choice(inliers,8)
        else:
            index = np.random.choice(N2, 8, replace = False)
            
    
    bestF = np.array(bestF)
    return bestF
    

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass


if __name__ == "__main__":
    
    # pts = np.load('../data/some_corresp.npz')
    # pts1 = pts["pts1"]
    # pts2 = pts["pts2"]
    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')
    
    # M = np.max(np.shape(im1))
    # F = eightpoint(pts1, pts2, M)
    # print("F = ", F)
    # helper.displayEpipolarF(im1, im2, F)
    # np.savez('q2_1.npz', F = F, M = M)
    
    # intrinsics = np.load('../data/intrinsics.npz')
    # K1 = intrinsics["K1"]
    # K2 = intrinsics["K2"]
    # E = essentialMatrix(F, K1, K2)
    # print("E = ", E)
    
     # helper.epipolarMatchGUI(im1, im2, F)
     
     #5.1
    pts = np.load('../data/some_corresp_noisy.npz')
    pts1 = pts["pts1"]
    pts2 = pts["pts2"]
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    M = np.max(np.shape(im1))
    # F = eightpoint(pts1, pts2, M)
    nIters=200
    tol=0.002
    args = []
    args.append((pts1,pts2,M, nIters, tol))
    
    p = Pool()
    F= p.map(ransacF , args)
    F = np.array(F).reshape(3,3)
    p.close()
    p.join()
    # F, inliers = ransacF(pts1, pts2, M)
    print("F = ", F)

    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics["K1"]
    K2 = intrinsics["K2"]
    E = essentialMatrix(F, K1, K2)
    print("E = ", E)
    
    helper.displayEpipolarF(im1, im2, F)