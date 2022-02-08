import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
import scipy.ndimage.morphology as morphology
import scipy.ndimage
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    # M = InverseCompositionAffine(image1, image2, threshold, num_iters)
   
    M = np.linalg.inv(M)
    image2_warp = scipy.ndimage.affine_transform(image2, M[0:2,0:2], M[0:2,2], output_shape = image1.shape)

    diff = abs(image1 - image2_warp)
    
    index = np.where((diff > tolerance) & (image1 != .0) & (image2_warp != .0))
    mask[index] = 0
    
    mask = morphology.binary_erosion(mask, structure = np.ones((2,2)), iterations=1)

    mask = morphology.binary_dilation(mask, iterations = 1)
    return mask
