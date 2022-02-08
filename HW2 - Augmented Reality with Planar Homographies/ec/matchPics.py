import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
    ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	

	#Convert Images to GrayScale
    I1gray = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
    I2gray = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)	
	
	#Detect Features in Both Images
    locs1 = corner_detection(I1gray, sigma)
    locs2 = corner_detection(I2gray, sigma)	
	#Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1gray, locs1)
    desc2, locs2 = computeBrief(I2gray, locs2)
    
	#Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2
