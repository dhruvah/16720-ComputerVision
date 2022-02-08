import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
import cv2
import glob
import os
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.30, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')

num_frames = seq.shape[2]

for frame in range(num_frames - 1):
    image1 = seq[:,:,frame]
    image2 = seq[:,:,frame + 1]
    
    mask = SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance)
    cars = np.where(mask == 0)
    
    # if frame == 30 or frame == 60 or frame == 90 or frame == 120:
    plt.figure()
    plt.axis('off')
    plt.imshow(image2, cmap = 'gray')
    plt.plot(cars[1], cars[0],'.',color = 'r')
    # plt.show()
    plt.savefig('../aerialseq/img{}.png'.format(frame), bbox_inches = "tight")
    
filename = '../result/aerialtracking.avi'

img_array = []
filenames = glob.glob("../aerialseq/*.png")
filenames.sort(key = os.path.getmtime)

for image in filenames:
        img = cv2.imread(image)
        h, w, l = img.shape
        size = (w, h)
        img_array.append(img)
        print(image)

writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'),15, size)
        
for i in range(len(img_array)):
    writer.write(img_array[i])

writer.release()
