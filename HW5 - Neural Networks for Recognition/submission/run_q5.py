import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

n_examples, dim = train_x.shape
# Q5.1 & Q5.2
# initialize layers here
initialize_weights(dim, hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'hidden1')
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
initialize_weights(hidden_size, dim, params, 'output')

params['m_Wlayer1'] = np.zeros((dim, hidden_size))
params['m_blayer1'] = np.zeros(hidden_size)
params['m_Whidden1'] = np.zeros((hidden_size, hidden_size))
params['m_bhidden1'] = np.zeros(hidden_size)
params['m_Whidden2'] = np.zeros((hidden_size, hidden_size))
params['m_bhidden2'] = np.zeros(hidden_size)
params['m_Woutput'] = np.zeros((hidden_size, dim))
params['m_boutput'] = np.zeros(dim)

epochs = np.arange(max_iters)
train_acc_data = []
valid_acc_data = []
train_loss_data = []
valid_loss_data = []
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        h = forward(xb, params, 'layer1', relu)
        h1 = forward(h, params, 'hidden1', relu)
        h2 = forward(h1, params, 'hidden2', relu)
        out = forward(h2, params, 'output', sigmoid)
    
        loss = np.sum(np.square(out - xb))
        total_loss += loss
        delta = 2*(out - xb) 
        delta1 = backwards(delta, params, 'output', sigmoid_deriv)
        delta2 = backwards(delta1, params, 'hidden2', relu_deriv)
        delta3 = backwards(delta2, params, 'hidden1', relu_deriv)
        backwards(delta3, params, 'layer1', relu_deriv)           
        
        params['m_Wlayer1'] = 0.9*params['m_Wlayer1'] - learning_rate*params['grad_Wlayer1']
        params['Wlayer1'] += params['m_Wlayer1']
        params['m_blayer1'] = 0.9*params['m_blayer1'] - learning_rate*params['grad_blayer1']
        params['blayer1'] += params['m_blayer1']
        
        params['m_Whidden1'] = 0.9*params['m_Whidden1'] - learning_rate*params['grad_Whidden1']
        params['Whidden1'] += params['m_Whidden1']
        params['m_bhidden1'] = 0.9*params['m_bhidden1'] - learning_rate*params['grad_bhidden1']
        params['bhidden1'] += params['m_bhidden1']
        
        params['m_Whidden2'] = 0.9*params['m_Whidden2'] - learning_rate*params['grad_Whidden2']
        params['Whidden2'] += params['m_Whidden2']
        params['m_bhidden2'] = 0.9*params['m_bhidden2'] - learning_rate*params['grad_bhidden2']
        params['bhidden2'] += params['m_bhidden2']
        
        params['m_Woutput'] = 0.9*params['m_Woutput'] - learning_rate*params['grad_Woutput']
        params['Woutput'] += params['m_Woutput']
        params['m_boutput'] = 0.9*params['m_boutput'] - learning_rate*params['grad_boutput']
        params['b_output'] += params['m_boutput']
    
    total_loss = total_loss/n_examples
    
    h_valid = forward(valid_x, params, 'layer1', relu)
    h1_valid = forward(h_valid, params, 'hidden1', relu)
    h2_valid = forward(h1_valid, params, 'hidden2', relu)
    out_valid = forward(h2_valid, params, 'output', sigmoid)
    loss_valid = np.sum(np.square(out_valid - valid_x))
    
    loss_valid = loss_valid / valid_x.shape[0]
    
    train_loss_data.append(total_loss)
    valid_loss_data.append(loss_valid)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
plt.figure()
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, train_loss_data, label = "Training Loss")
plt.plot(epochs, valid_loss_data, label = "Validation Loss")
plt.legend()
plt.show()

h_valid = forward(valid_x, params, 'layer1', relu)
h1_valid = forward(h_valid, params, 'hidden1', relu)
h2_valid = forward(h1_valid, params, 'hidden2', relu)
out_valid = forward(h2_valid, params, 'output', sigmoid)

for k in range(5):
    
    i = 300*k
    plt.subplot(2,2,1)
    plt.imshow(valid_x[i,:].reshape(32,32).T)
    plt.subplot(2,2,2)
    plt.imshow(out_valid[i,:].reshape(32,32).T)
    plt.subplot(2,2,3)
    plt.imshow(valid_x[i+1,:].reshape(32,32).T)
    plt.subplot(2,2,4)
    plt.imshow(out_valid[i+1,:].reshape(32,32).T)
    plt.show()
# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
totalpsnr = 0
for i in range(valid_x.shape[0]):
    psnr = skimage.measure.compare_psnr(valid_x[i,:], out_valid[i,:])
    totalpsnr += psnr
    
avgpsnr = totalpsnr/valid_x.shape[0]
print("Avg PSNR = ", avgpsnr)