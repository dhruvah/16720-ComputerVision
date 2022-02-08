import numpy as np
import cv2
#Import necessary functions
from loadVid import loadVid
from opts import get_opts
from planarH import computeH_ransac
from planarH import compositeH
from matchPics import matchPics
from helper import plotMatches
from matplotlib import pyplot as plt
from multiprocessing import Pool
import glob

def createVideoFiles(args):
    opts, numframes, ar_source, book, cv_cover, bookdim = args
    
    for i in range(numframes):
        bookframe = book[i,:,:,:]
        pandaframe = ar_source[i,:,:]
        panda_crop = pandaframe[40:320, 208:431, :]
        panda = cv2.resize(panda_crop, bookdim, interpolation = cv2.INTER_AREA)
        
        matches, locs1, locs2 = matchPics(cv_cover, bookframe, opts)
        
        # plotMatches(cv_cover, bookframe, matches, locs1, locs2)
        locs1 = locs1[matches[:,0]]
        locs2 = locs2[matches[:,1]]
        locs1[:,[1,0]] = locs1[:,[0,1]]
        locs2[:,[1,0]] = locs2[:,[0,1]]
        
        
        H2to1, inliers = computeH_ransac(locs1, locs2, opts)
    
        image = compositeH(H2to1, panda, bookframe)
        np.save("../test1/img{}.npy".format(i), image)

if __name__ == '__main__':
#Write script for Q3.1
    ar_source = loadVid('../data/ar_source.mov')
    book = loadVid('../data/book.mov')
    
    opts = get_opts()
    # numframes = 10
    # numframes = 108
    numframes = np.size(ar_source, axis = 0)
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    
    w = cv_cover.shape[1]
    h = cv_cover.shape[0]
    bookdim = (w,h)
    
    fileName = '../result/ar4.avi'
    vidsize = (640,480)
    writer = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc(*"MJPG"),25.0, vidsize)
    
    
    args = []
    args.append((opts,numframes,ar_source,book, cv_cover,bookdim))
    print("iter")
    p = Pool()
    p.map(createVideoFiles , args)
    
    p.close()
    p.join()
    
    for image in glob.glob("../test1/*.npy"):
        img = np.load(image)
        writer.write(img)















    # plt.imshow(image)
    