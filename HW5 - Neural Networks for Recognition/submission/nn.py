import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    
    lowerbound = -np.sqrt(6/(in_size + out_size))
    upperbound = -lowerbound
    
    W = np.random.uniform(lowerbound, upperbound, (in_size, out_size))
    b = np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):

    res = 1 / (1 + np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    pre_act = X @ W + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None
    examples, classes = np.shape(x)
    x_max = np.max(x, axis = 1)
    c = -x_max.reshape(examples, 1)
    res = np.exp(x+c) / np.sum(np.exp(x+c), axis = 1).reshape(examples,1)
    
    # res = np.zeros((examples, classes))
    # for i in range(examples):
    #     xi = x[i,:]
    #     x_max = np.max(xi)
        
    #     res[i,:] = np.exp(xi - x_max) / np.sum(np.exp(xi-x_max))
        
    # c = np.expand_dims(-x_max, axis = 1)
    # res = np.exp(x+c)/ np.expand_dims(np.sum(np.exp(x+c), axis = 1), axis = 1)
    
    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    
    examples, classes = np.shape(y)
    loss = -np.sum(y * np.log(probs))
    
    count = 0
    for i in range(examples):
        idx = np.argmax(probs[i,:])
        if(y[i,idx] == 1):
            count += 1
            
    acc = count/examples
    
    # true = np.where(y[np.where(probs ==1)] == 1, 1, 0)
    # acc = np.sum(true) / examples

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    derivative = activation_deriv(post_act)
    examples, classes = np.shape(delta)
    
    grad_W = X.T @ (derivative * delta)
    grad_X = (derivative * delta) @ W.T
    grad_b = (np.ones((1, examples))) @ (derivative * delta)
    grad_b = grad_b.flatten()
    
    
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    
    examples = x.shape[0]
    idx = np.random.choice(examples, size = (int(examples/batch_size), batch_size))
    
    for i in range(len(idx)):
        batch_x = x[idx[i], :]
        batch_y = y[idx[i], :]
        batches.append((batch_x, batch_y))
        
    return batches
