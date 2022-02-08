import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.006
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

n_examples, dim = train_x.shape
n_examples, classes = train_y.shape
epochs = np.arange(max_iters)
train_acc_data = []
valid_acc_data = []
train_loss_data = []
valid_loss_data = []

model = torch.nn.Sequential(
    torch.nn.Linear(dim, hidden_size), 
    torch.nn.Sigmoid(), 
    torch.nn.Linear(hidden_size, classes))
    # torch.nn.Softmax())

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

valid_x = torch.from_numpy(valid_x).float()
valid_y = torch.from_numpy(valid_y).int()

for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    
    for xb, yb in batches:
        xb = torch.from_numpy(xb).float()
        yb = torch.from_numpy(yb).int()
        
        label = np.where(yb == 1)[1]
        label = torch.tensor(label)
        
        out = model(xb)
        
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        _, predicted = torch.max(out.data, 1)
        
        total_acc += ((label == predicted).sum().item())
        total_loss += loss
    
    total_acc = total_acc/ n_examples
    total_loss = total_loss / batch_num
    
    
    out_valid = model(valid_x)
    label_valid = torch.tensor(np.where(valid_y == 1)[1])
    loss_valid = criterion(out_valid, label_valid)
    
    _, predicted_valid = torch.max(out_valid.data,1)
    
    acc_valid = (label_valid == predicted_valid).sum().item()
    acc_valid /= valid_x.shape[0]
    
    train_acc_data.append(total_acc)
    valid_acc_data.append(acc_valid)
    train_loss_data.append(total_loss)
    valid_loss_data.append(loss_valid)    
    
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

print("acc = ", total_acc)
print('Validation accuracy: ',acc_valid)

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









     
