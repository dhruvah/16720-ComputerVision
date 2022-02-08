import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-1, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
# template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

girlseqrects = np.array(rect)
num_frames = seq.shape[2]


for frame in range(num_frames - 1):
    
    It = seq[:,:,frame]
    It1 = seq[:,:,frame+1]
    
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    
    rect = rect + np.array([p[0], p[1], p[0], p[1]])
    girlseqrects = np.vstack((girlseqrects, rect))
    
    if frame ==1 or frame ==20 or frame == 40 or frame == 60 or frame == 80:
        fig, ax = plt.subplots()
        plt.axis('off')
        ax.imshow(It1, cmap = 'gray')
        patch = patches.Rectangle((rect[0],  rect[1]),
                                  (rect[2] - rect[0]),
                                  (rect[3] - rect[1]),
                                  linewidth = 1, edgecolor = 'r',
                                  facecolor = 'none')
        ax.add_patch(patch)
        plt.show()
        
        
np.save("../result/girlseqrects.npy", girlseqrects)