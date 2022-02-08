import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH
from helper import plotMatches
from matplotlib import pyplot as plt

#Import necessary functions



#Write script for Q2.2.4
opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

plotMatches(cv_cover, cv_desk, matches, locs1, locs2)
locs1 = locs1[matches[:,0]]
locs2 = locs2[matches[:,1]]
locs1[:,[1,0]] = locs1[:,[0,1]]
locs2[:,[1,0]] = locs2[:,[0,1]]

print(np.shape(locs1))

H2to1, inliers = computeH_ransac(locs1, locs2, opts)


template = cv2.resize(hp_cover, (cv_cover.shape[1],cv_cover.shape[0]), interpolation = cv2.INTER_AREA)

image = compositeH(H2to1, template, cv_desk)

plt.imshow(image)
plt.axis('off')