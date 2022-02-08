import numpy as np
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
    
    n = 4 #number of points to compute homography
    A = np.zeros((8,9))
    N = x1.shape[0]
    
    # index = np.random.choice(N, n, replace = False)
    # X1 = x1[index]
    # X2 = x2[index]
    
    u = x1[:,0].reshape(n,1)
    v = x1[:,1].reshape(n,1)
    x = x2[:,0].reshape(n,1)
    y = x2[:,1].reshape(n,1)
    
    A1 = np.hstack((-1*x, -1*y,-1*np.ones((n,1)), np.zeros((n,3)), np.multiply(u,x), np.multiply(u,y), u))
    A2 = np.hstack((np.zeros((n,3)), -1*x, -1*y, -1*np.ones((n,1)), np.multiply(v,x), np.multiply(v,y), v))

    A = np.vstack((A1, A2))
    U,S,VT = np.linalg.svd(A)
    
    h = VT[-1,:]
    
    H2to1 = h.reshape(3,3)

    return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
    N = x1.shape[0]
    C1 = np.mean(x1, axis = 0)
    C2 = np.mean(x2, axis = 0)

	#Shift the origin of the points to the centroid
    x1 = x1 - C1
    x2 = x2 - C2

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    
    s1 = np.sqrt(2)/(np.mean(np.sqrt(np.sum(np.square(x1),axis = 1))))
    s2 = np.sqrt(2)/(np.mean(np.sqrt(np.sum(np.square(x2),axis = 1))))   
    x1_norm = s1*x1;
    x2_norm = s2*x2;

	#Similarity transform 1
    T1 = np.array([[s1,0,-s1*C1[0]], [0,s1,-s1*C1[1]], [0,0,1]])
	#Similarity transform 2
    T2 = np.array([[s2,0,-s2*C2[0]], [0,s2,-s2*C2[1]], [0,0,1]])


	#Compute homography
    H = computeH(x1_norm, x2_norm)

	#Denormalization
    T1inv = np.linalg.inv(T1)
    H2to1 = T1inv @ H @ T2

    return H2to1

    
    
def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    N1 = np.size(locs1[:,0])
    N2 = np.size(locs2[:,0])
    

    inliers = np.zeros(N1)
    maxnum = -1
    bestH2to1 = None
    
    x1_homo = np.hstack((locs1, np.ones(N1).reshape(N1,1)))
    x2_homo = np.hstack((locs2, np.ones(N2).reshape(N2,1)))
    
    for i in range(max_iters):
        index = np.random.choice(N2, 4, replace = False)
        x1 = locs1[index]
        x2 = locs2[index]
        
        H = computeH_norm(x1,x2)
        x1_pred_homo = H @ x2_homo.T
        scale = x1_pred_homo[-1,:]
        x1_pred_homo = x1_pred_homo/scale
        x1_pred_homo = x1_pred_homo.T
        diff = (x1_pred_homo - x1_homo)[:,[0,1]]
        normdiff = np.linalg.norm(diff, axis = 1)
        inlier = np.where(normdiff < inlier_tol , 1, 0)
        inliers_index = np.where(normdiff < inlier_tol)
        
        num_inliers = np.sum(inlier)
        
        if num_inliers > maxnum:
            maxnum = num_inliers
            bestH2to1 = H
            inliers = inlier
        
    return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
    H = np.linalg.inv(H2to1)
    
    dim_img = (img.shape[1], img.shape[0]) 
    dim_template = (template.shape[0],template.shape[1], template.shape[2])
    
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
	#Create mask of same size as template
    mask = 255*np.ones(dim_template)
	#Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H, dim_img)
    #Warp template by appropriate homography
    warped_temp = cv2.warpPerspective(template, H, dim_img)
    warped_temp = cv2.cvtColor(warped_temp, cv2.COLOR_BGR2RGB)
	#Use mask to combine the warped template and the image
    np.putmask(img, warped_mask, warped_temp)
    composite_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    return composite_img


