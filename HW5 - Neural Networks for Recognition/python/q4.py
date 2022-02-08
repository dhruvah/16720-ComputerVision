import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
from skimage.filters import threshold_otsu

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    denoise_image = skimage.restoration.denoise_bilateral(image, multichannel=True)
    greyscale_image = skimage.color.rgb2grey(denoise_image)
    threshold = threshold_otsu(greyscale_image)
    bw = skimage.morphology.closing(greyscale_image < threshold, skimage.morphology.square(10))
    cleared = skimage.segmentation.clear_border(bw)
    label_image = skimage.measure.label(cleared,background=0,connectivity = 2)
    
    for region in skimage.measure.regionprops(label_image):
        if region.area >= 200:
            bboxes.append(region.bbox)
        

    return bboxes, bw