import numpy as np
import cv2
from matchPics import matchPics
import scipy
from opts import get_opts
from helper import plotMatches
import matplotlib.pyplot as plt

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
matchesnumber = np.zeros((36))

for i in range(36):
	#Rotate Image
    cv_cover_rot = scipy.ndimage.rotate(cv_cover, 10*i)
    
	#Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(cv_cover, cv_cover_rot, opts)
	#Update histogram
    matchesnumber[i] = np.size(matches, axis = 0)
    
    if(i == 3):
        plotMatches(cv_cover, cv_cover_rot, matches, locs1, locs2)
    if(i == 15):
        plotMatches(cv_cover, cv_cover_rot, matches, locs1, locs2)
    if(i == 30):
        plotMatches(cv_cover, cv_cover_rot, matches, locs1, locs2)
        
#Display histogram

# plt.hist(matchesnumber, bins = 36)
# plt.show()
# x = np.arange(36)
# plt.xlabel("Rotation iteration")
# plt.ylabel("No. of matches")
# plt.bar(x,matchesnumber)
