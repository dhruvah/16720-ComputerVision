import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.005
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
n_input = train_x.shape[1]
n_output = train_y.shape[1]
initialize_weights(n_input, hidden_size, params, 'layer1')
initialize_weights(hidden_size, n_output, params, 'output')

n_examples = train_x.shape[0]

epochs = np.arange(max_iters)
train_acc_data = []
valid_acc_data = []
train_loss_data = []
valid_loss_data = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc
        
        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient

        params['Wlayer1'] -= learning_rate*params['grad_Wlayer1']
        params['blayer1'] -= learning_rate*params['grad_blayer1']
        params['Woutput'] -= learning_rate*params['grad_Woutput']
        params['boutput'] -= learning_rate*params['grad_boutput']

    total_acc = total_acc / batch_num
    total_loss = total_loss/n_examples
        
    h1_valid = forward(valid_x, params, 'layer1')
    probs_valid = forward(h1_valid, params, 'output', softmax)
    loss_valid, acc_valid = compute_loss_and_acc(valid_y, probs_valid)
    loss_valid = loss_valid / valid_x.shape[0]
    
    train_acc_data.append(total_acc)
    valid_acc_data.append(acc_valid)
    train_loss_data.append(total_loss)
    valid_loss_data.append(loss_valid)    
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

print("acc = ", total_acc)
##Plot
plt.figure()
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(epochs, train_acc_data, label = "Training Accuracy")
plt.plot(epochs, valid_acc_data, label = "Validation Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, train_loss_data, label = "Training Loss")
plt.plot(epochs, valid_loss_data, label = "Validation Loss")
plt.legend()
plt.show()


# run on validation set and report accuracy! should be above 75%
valid_acc = None
h1_valid = forward(valid_x, params, 'layer1')
probs_valid = forward(h1_valid, params, 'output', softmax)
_, valid_acc = compute_loss_and_acc(valid_y, probs_valid)
 
print('Validation accuracy: ',valid_acc)

test_acc = None
h1_test = forward(test_x, params, 'layer1')
probs_test = forward(h1_test, params, 'output', softmax)
_, test_acc = compute_loss_and_acc(test_y, probs_test)

print('Test accuracy: ', test_acc)

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
W_firstlayer = params['Wlayer1']
fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols = (8,8), axes_pad= 0.0)
for i in range(W_firstlayer.shape[1]):
    grid[i].imshow(W_firstlayer[:,i].reshape(32,32))

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
h1_test = forward(test_x, params, 'layer1')
probs_test = forward(h1_test, params, 'output', softmax)
_, acc_test = compute_loss_and_acc(test_y, probs_test)

valid_y_pred = np.argmax(probs_test, axis = 1)
for i in range(test_y.shape[0]):
    pred = valid_y_pred[i]
    label = np.argmax(test_y[i])
    confusion_matrix[label][pred] +=1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()