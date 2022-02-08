# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

# Imports
import numpy as np
from skimage.io import imread
from skimage.color import rgb2xyz
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.sparse
from utils import integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centered on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the sphere in an array of size (3,)

    rad : float
        The radius of the sphere

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the sphere
    """
    W, H = res
    image = np.ones((H,W))
    radiuspixel = int(rad/pxSize) 
    
    xc, yc, zc = center
    xcpixel = abs(int(xc - W//2))
    ycpixel = abs(int(yc - H//2))
    l = (light/pxSize).astype(int)
    l = l - center
    l[1] = -l[1]
    for x in range(W):
        for y in range(H):
            
            dist = radiuspixel - np.sqrt((xcpixel - x)**2 + (ycpixel - y)**2)
            if dist >= 0:
                Nx = ((x - xcpixel)/radiuspixel)
                Ny = ((y - ycpixel)/radiuspixel)
                Nz = (np.sqrt(1 - Nx**2 - Ny**2))
            
                n = np.array([Nx, Ny, Nz])
                I = n @ l
                image[y,x] = I
            
            
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    img = imread('../data/input_1.tif')
    H, W, _ = img.shape
    s = (H, W)
    I = np.zeros((7, H*W))
    for i in range(7):
        img = imread('../data/input_{}.tif'.format(i+1))
        img = rgb2xyz(img)
        img = img[:,:,1].flatten()
        I[i,:] = img
        
    L = np.load('../data/sources.npy')
    L = L.T
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    _, P = I.shape
    B = np.zeros((3,P))
    
    #tried the sparse module to create a giant box diagnoal matrix but ran into
    #memory problems. Tried to create a gian identity matrix and kron function
    #also, but still memory problems. So went for the loop.
    for i in range(P):
        A = L.T
        y = I[:,i]
        x, _, _, _ = scipy.sparse.linalg.lsqr(A, y)[:4]
        B[:,i] = x
        
    
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B, axis = 0)
    
    normals = B/albedos
    
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    
    H, W = s
    albedoIm = albedos.reshape(s)
    normalIm = normals.T.reshape(H, W, 3)

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    
    H, W = s
    normals = normals.T.reshape(H, W, 3)
    dfx = -normals[:,:,0]/normals[:,:,2]
    dfy = -normals[:,:,1]/normals[:,:,2]
    surface = integrateFrankot(dfx, dfy)
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    H, W = surface.shape
    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)
    X, Y = np.meshgrid(x, y)
    z = surface
    
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    # ax.set_zlim3d(-100, 100)
    surface = ax.plot_surface(X, Y, -z, cmap = 'coolwarm')
    # ax.view_init(55, 90)
    plt.show()  
    
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    surface = ax.plot_surface(X, Y, -z, cmap = 'coolwarm')
    ax.view_init(55, 90)
    plt.show()  
    
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    # ax.set_zlim3d(-100, 100)
    surface = ax.plot_surface(X, Y, -z, cmap = 'coolwarm')
    ax.view_init(50, 10)
    plt.show()  
    
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    surface = ax.plot_surface(X, Y, -z, cmap = 'coolwarm')
    ax.view_init(100, 90)
    plt.show()  

if __name__ == '__main__':

    #1 a)
    center = np.array([0,0,0])
    rad = 0.75
    light1 = np.array([1,1,1])/np.sqrt(3)
    pxSize = 7e-4
    res = [3840, 2160]
    img1 = renderNDotLSphere(center, rad, light1, pxSize, res);
    plt.imshow(img1, cmap = 'gray')
    plt.show()
    
    light2 = np.array([1,-1,1])/np.sqrt(3)
    img2 = renderNDotLSphere(center, rad, light2, pxSize, res);
    plt.imshow(img2, cmap = 'gray')
    plt.show()
    
    light3 = np.array([-1,-1,1])/np.sqrt(3)
    img3 = renderNDotLSphere(center, rad, light3, pxSize, res);
    plt.imshow(img3, cmap = 'gray')
    plt.show()
    
    # 1b)
    
    I, L, s = loadData()
    
    B = estimatePseudonormalsCalibrated(I, L)
    
    albedos, normals = estimateAlbedosNormals(B)
    
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
    # normalizing Image
    normnormalIm = (normalIm - np.min(normalIm)) / (np.max(normalIm) - np.min(normalIm))
    
    plt.figure()
    plt.imshow(albedoIm, cmap = 'gray')
    plt.show()
    
    plt.figure()
    plt.imshow(normalIm, cmap= 'rainbow')
    plt.show()
    
    plt.figure()
    plt.imshow(normnormalIm, cmap = 'rainbow')
    plt.show()
    
    surface = estimateShape(normals, s)
    plotSurface(surface)
