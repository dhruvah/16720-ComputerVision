import os
import multiprocessing as mp
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import scipy.spatial.distance
import skimage.color
import sklearn.cluster
from scipy.ndimage import gaussian_filter
from tempfile import TemporaryFile


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    
    filter_scales = opts.filter_scales
    n_scale = len(opts.filter_scales)
    
    if img.ndim == 2:
        img = np.dstack((img, img, img))


    [H, W, channel] = np.shape(img)
    
    if channel == 1: #greyscale
        img = np.matlib.repmat(img,1,1,3)
        
    channel = 3
    
    img = skimage.color.rgb2lab(img)
  #  img_r = img[:,:,0]
  #  img_g = img[:,:,1]
  #  img_b = img[:,:,2]
    F = 4*n_scale
    
    filter_responses = np.zeros((np.size(img, 0), np.size(img,1), 3*F), dtype ="uint8")
    gauss = np.zeros(img.shape, dtype ="uint8")
    lapgauss = np.zeros(img.shape, dtype ="uint8")
    imgx = np.zeros(img.shape, dtype ="uint8")
    imgy = np.zeros(img.shape, dtype ="uint8")
    
    for i in range(n_scale):
        for j in range(3):
            
            
            gauss[:,:,j] = scipy.ndimage.gaussian_filter(img[:,:,j], filter_scales[i], output=np.float64)
            lapgauss[:,:,j] = scipy.ndimage.gaussian_laplace(img[:,:,j], filter_scales[i], output=np.float64)
            imgx[:,:,j] = scipy.ndimage.gaussian_filter(img[:,:,j], filter_scales[i], [0,1], output=np.float64)
            imgy[:,:,j] = scipy.ndimage.gaussian_filter(img[:,:,j], filter_scales[i], [1,0], output=np.float64)
            
            
            
        filter_responses[:, :, 12*i:12*(i+1)] = np.dstack((gauss, lapgauss, imgx, imgy))
    

    return filter_responses
    
    

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    
    
    
    opts,i,alpha,image_path = args
    
    out_path = os.path.join(opts.feat_dir, "{}.npy".format(i))
    
    img = Image.open(image_path)
    img = np.array(img).astype(np.float32)/255
  
 
#    img = Image.open(img_path)
#    img = np.array(img).astype(np.float32)/255
    response = extract_filter_responses(opts, img)
    alpha_responses = np.random.permutation(response.reshape(img.shape[0]*img.shape[1], -1))[:alpha]
    
    np.save(out_path, alpha_responses)
    
    return

    

def compute_dictionary(opts, n_worker=8):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    n_scale = len(opts.filter_scales)
    F = 4*n_scale
    


    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    T = len(train_files)

    alpha = opts.alpha
    
    pool = mp.Pool(n_worker)
    
    args_list =[]  
    for i in range(T):
        args_list.append((opts,i,alpha,os.path.join("../data",train_files[i])))
        

    pool.map(compute_dictionary_one_image, args_list) 
    
    filter_responses = []

    for i in range(T):
        filter_responses.append(np.load(os.path.join(opts.feat_dir,"{}.npy".format(i))))
                                
    
    filter_responses = np.array(filter_responses).reshape(-1,3*F)


    print("Now Running Kmeans.....")

    
    #kmeans
    kmeans = sklearn.cluster.KMeans(n_clusters = K, n_jobs=-1).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    
    print("Finished!")
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    return
    
    

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    filter_response = extract_filter_responses(opts, img)
    H = filter_response.shape[0]
    W = filter_response.shape[1]
    filter_response = filter_response.reshape(H*W,-1)
    dist = scipy.spatial.distance.cdist(filter_response,dictionary,'euclidean')
    wordmap = np.argmin(dist, axis = 1).reshape(H,W)
    return wordmap    
    

