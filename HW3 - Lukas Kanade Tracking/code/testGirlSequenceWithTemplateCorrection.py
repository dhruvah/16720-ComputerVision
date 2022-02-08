import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import cv2
import glob
import os

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

rect0 = rect

girlseqrects_wcrt = np.array(rect)
num_frames = seq.shape[2]

p0 = np.zeros(2)

T1x = seq[:,:,0]
It = seq[:,:,0]
for frame in range(num_frames - 1):
    
    It1 = seq[:,:,frame+1]
    
    p = LucasKanade(It, It1, rect, threshold, num_iters, p0)
    p_n = p + [rect[0] - rect0[0], rect[1] - rect0[1]]
    
    pstar_n = LucasKanade(T1x, It1, rect0, threshold, num_iters, p_n)
    
    if np.linalg.norm(p_n - pstar_n) < template_threshold:
        pstar = pstar_n - [rect[0] - rect0[0], rect[1] - rect0[1]]
        
        rect = rect + np.array([pstar[0], pstar[1], pstar[0], pstar[1]])
        girlseqrects_wcrt = np.vstack((girlseqrects_wcrt, rect))
        It = seq[:,:,frame+1]
        p0 = np.zeros(2)
        
    else:
        rect = rect + np.array([p[0], p[1], p[0], p[1]])
        girlseqrects_wcrt = np.vstack((girlseqrects_wcrt, rect))
        p0 = p
    

# frameindex = [1, 20, 40, 60, 80]    
girlseqrects = np.load("../result/girlseqrects.npy")

for i in range(num_frames):
    rect = girlseqrects[i,:]
    rect_wcrt = girlseqrects_wcrt[i,:]
    

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(seq[:,:,i], cmap = 'gray')
    # patch1 = patches.Rectangle((rect[0],  rect[1]),
    #                            (rect[2] - rect[0]),
    #                            (rect[3] - rect[1]),
    #                            linewidth = 1, edgecolor = 'r',
    #                            facecolor = 'none')
    
    patch2 = patches.Rectangle((rect_wcrt[0],  rect_wcrt[1]),
                               (rect_wcrt[2] - rect_wcrt[0]),
                               (rect_wcrt[3] - rect_wcrt[1]),
                               linewidth = 1, edgecolor = 'r',
                               facecolor = 'none')
    # ax.add_patch(patch1)
    ax.add_patch(patch2)
    plt.savefig('../girlseq/img{}.png'.format(i), bbox_inches = "tight")
    # plt.show()

filename = '../result/girltracking.avi'

img_array = []
filenames = glob.glob("../girlseq/*.png")
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
# np.save("../result/girlseqrects-wcrt.npy",girlseqrects_wcrt)