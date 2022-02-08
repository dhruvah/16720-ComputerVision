import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(1-bw, cmap = "gray")
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    
    boxes = [[]]
    row = 0
    bboxes.sort(key = lambda x:x[2])
    letter_base = bboxes[0][2]
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        if (minr >= letter_base + 10):
            row +=1;
            boxes.append([])
            letter_base = maxr
            # print(letter_base)
        boxes[row].append(bbox)
    
    
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    
    for row in boxes:
        row.sort(key = lambda x:x[1])
        output = ""
        for box in row:
            minr, minc, maxr, maxc = box
            letter = bw[minr:maxr, minc:maxc]
            if (img == '01_list.jpg'):
                pad_size = 15
                square_size = 3
            if (img == '02_letters.jpg'):
                pad_size = 40
                square_size = 3
            if (img == '03_haiku.jpg'):
                pad_size = 30
                square_size = 2
            if (img == '04_deep.jpg'):
                pad_size = 50
                square_size = 3
                
            letter = np.pad(letter, ((pad_size,pad_size), (pad_size,pad_size)), 'constant', constant_values = 0.0)
            letter = skimage.transform.resize(letter, (32, 32))
            letter = skimage.morphology.dilation(letter, skimage.morphology.square(square_size))
            # letter = skimage.morphology.erosion(letter, np.array([[0,1,0],[1,1,1],[0,1,0]]))
            
            letter = 1 - letter
            
           
                
            # plt.imshow(letter)
            # plt.show()
            letter = letter.T
            
            x = letter.reshape(1,-1)
            
            h1 = forward(x, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            index = np.argmax(probs[0,:])
            output += (letters[index] + " ")
        
        print(output)
    
    print("______________________________________________________________________")
    ##########################
    ##### your code here #####
    ##########################
    