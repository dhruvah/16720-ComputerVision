import numpy as np
import cv2
#Import necessary functions
from opts.py import get_opts
from planarH import computeH_ransac
from planarH import compositeH
from matchPics import matchPics
from helper import plotMatches
from matplotlib import pyplot as plt

opts = get_opts()

#Write script for Q4.2x
pano_left = cv2.imread('../IMG-7543.jpg')
pano_right = cv2.imread('../IMG-7544.jpg')

H1 = np.size(pano_left, axis = 0)
W1 = np.size(pano_left, axis = 1)
H2 = np.size(pano_right, axis = 0)
W2 = np.size(pano_right, axis = 1)

matches, locs1, locs2 = matchPics(pano_left, pano_right, opts)

plotMatches(pano_left, pano_right, matches, locs1, locs2)
locs1 = locs1[matches[:,0]]
locs2 = locs2[matches[:,1]]
locs1[:,[1,0]] = locs1[:,[0,1]]
locs2[:,[1,0]] = locs2[:,[0,1]]

H2to1, inliers = computeH_ransac(locs1, locs2, opts)


corners_right = np.array([[0,0,W2-1,W2-1],[0,H2-1,H2-1,0],[1,1,1,1]])

corners_left_pred = H2to1 @ corners_right

corners_left_pred = (corners_left_pred/corners_left_pred[-1,:]).round().astype(int)

Hmax = np.amax([H1,np.amax(corners_left_pred[1,:])])
Wmax = np.amax([W1,np.amax(corners_left_pred[0,:])])
Hmin = np.amin([0, np.amin(corners_left_pred[1,:])])
Wmin = np.amin([0, np.amin(corners_left_pred[0,:])])
    
Hpred = Hmax - Hmin
Wpred = Wmax - Wmin

M = np.array([[1.0,0.0,0.0],[0.0,1.,-Hmin],[0.0,0.0,1.0]])
print(M)
warp_pano_left = cv2.warpPerspective(pano_left, M, (Wpred,Hpred))
warp_pano_right = cv2.warpPerspective(pano_right, M @ H2to1, (Wpred,Hpred))

panorama = np.maximum(warp_pano_right,warp_pano_left)


panorama = np.uint8(panorama)
panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
plt.imshow(panorama)    
cv2.imwrite('../result/panorama.jpg', panorama)
plt.axis('off')