# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt
import scipy.sparse

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    U, S, Vt = np.linalg.svd(I, full_matrices = False)
    
    S[3:7] = 0
    S = np.diag(S)
    
    L = U @ S**(1/2)
    L = L[:, 0:3]
    L = L.T
    
    B = S**(1/2) @ Vt
    B = B[0:3,:]

    return B, L

if __name__ == "__main__":

   I, L, s = loadData()
   
   B, Lest = estimatePseudonormalsUncalibrated(I)
   albedos, normals = estimateAlbedosNormals(B)
   albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
   #normalizing Image
   normnormalIm = (normalIm - np.min(normalIm)) / (np.max(normalIm) - np.min(normalIm))
    
    #2a
   plt.figure()
   plt.imshow(albedoIm, cmap = 'gray')
   plt.show()
    
   plt.figure()
   plt.imshow(normalIm, cmap= 'rainbow')
   plt.show()

   plt.figure()
   plt.imshow(normnormalIm, cmap = 'rainbow')
   plt.show()

   #2c
   plt.figure()
   plt.imshow(L, cmap = 'hot')
   plt.figure()
   plt.imshow(Lest, cmap = 'hot')
   plt.show()
   
   #2d
   surface = estimateShape(normals, s)
   plotSurface(surface)
   
   # #2e
   Bt = enforceIntegrability(B, s)
   albedost, normalst = estimateAlbedosNormals(Bt)
   surfacet = estimateShape(normalst, s)
   # plotSurface(surfacet)
   
   #2f
   #Bas Relief
   mu = 0
   v = 0
   l = 0.000001
   
   G = np.array([[1,0,0],[0,1,0],[mu,v,l]])
   
   Bbasr = (np.linalg.inv(G)).T @ Bt
   albedosbas, normalsbas = estimateAlbedosNormals(Bbasr)
   surfacebas = estimateShape(normalsbas, s)
   plotSurface(surfacebas)
   
   albedosbas, normalsbas = estimateAlbedosNormals(Bbasr)
   albedoImbas, normalImbas = displayAlbedosNormals(albedosbas, normalsbas, s)
   normnormalImbas = (normalImbas - np.min(normalImbas)) / (np.max(normalImbas) - np.min(normalImbas))
   
   # plt.figure()
   # plt.imshow(albedoIm, cmap = 'gray')
   # plt.show()
    
   # plt.figure()
   # plt.imshow(normnormalIm, cmap = 'rainbow')
   # plt.show()