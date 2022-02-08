import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K

    hist, bin_edges = np.histogram(wordmap.reshape(-1),bins = range(K+1), density = True)

    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    
    H = wordmap.shape[0]
    W = wordmap.shape[1]
    
    weights = []
    hist_all = []
    
    for l in range(L):
        if l == 0 or l ==1:
            weights.append(2**(-(L-1)))
        else:
            weights.append(2**(l-L))
            
    #top to bottom        
    for i in range(len(weights)):
        layer = len(weights)-1-i
        weight = weights[len(weights)-1-i]
        sh = int(H/(2**layer))
        sw = int(W/(2**layer))
        
        for r in range(2**layer):
            for c in range(2**layer):
                subword = wordmap[sh*r:sh*(r+1),sw*c:sw*(c+1)]
                hist = get_feature_from_wordmap(opts, subword)
                
                hist_all = np.hstack([hist_all,hist*weight])
                
    hist_all = hist_all / hist_all.sum()
    return hist_all
  
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    
    return feature   

def get_image_feature_call(args):
    i, opts, img_path, dictionary = args
    feature = get_image_feature(opts, img_path, dictionary)
    return feature
    

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    
    N = len(train_files)
         
    args_list=[]
    for i in range(N):
        img_path = os.path.join("../data",train_files[i])
        args_list.append((i,opts,img_path,dictionary))
        
    pool = multiprocessing.Pool(n_worker)

        
    features = np.array(pool.map(get_image_feature_call, args_list)).reshape(-1, K*(4**SPM_layer_num-1)//3)
   
    
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
    features=features,
    labels=train_labels,
    dictionary=dictionary,
    SPM_layer_num=SPM_layer_num)
    
    print("Trained system build complete")
   
    
    
    # ----- TODO -----
    pass

    ## example code snippet to save the learned system
    # np.savez_compressed(join(out_dir, 'trained_system.npz'),
    #     features=features,
    #     labels=train_labels,
    #     dictionary=dictionary,
    #     SPM_layer_num=SPM_layer_num,
    # )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    
    return np.sum(np.minimum(word_hist,histograms),axis=1)

    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']
    
    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    
    args_list = []
    
    
    
    for i in range(len(test_files)):
        img_path = os.path.join("../data",test_files[i])
        args_list.append((i,test_opts,img_path,dictionary,trained_features,trained_labels))
        
    pool = multiprocessing.Pool()
    
    
    C = np.zeros((8,8))
    
    pred = pool.map(evaluate_image, args_list)


    for i in range(len(test_files)):
        predlabel = pred[i]
        label = test_labels[i]
        C[label][predlabel] += 1.0
        
    np.save("confusion.npy",C)
    print(C)
    accuracy = np.trace(C) / np.sum(C)
    print("Accuracy: {}".format(accuracy))
    return C, accuracy

def evaluate_image(args):
    
    
    i, opts, img_path, dictionary, trained_features, trained_labels  = args
    
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    similarity = distance_to_set(trained_features, feature)
    pred_label = trained_labels[np.argmax(similarity)]
    return pred_label
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    