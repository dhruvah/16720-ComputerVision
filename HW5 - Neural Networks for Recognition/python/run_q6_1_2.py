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

max_iters = 50
# pick a batch size, learning rate
batch_size = 100
learning_rate = 0.006
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

n_examples, dim = train_x.shape
n_examples, classes = train_y.shape
n_valid_examples = valid_x.shape[0]
epochs = np.arange(max_iters)
train_acc_data = []
valid_acc_data = []
train_loss_data = []
valid_loss_data = []


train_x = torch.from_numpy(train_x).float()
valid_x = torch.from_numpy(valid_x).float()
valid_y = torch.from_numpy(valid_y).int()

label = np.where(train_y == 1)[1]
label = torch.tensor(label)
valid_label = np.where(valid_y == 1)[1]
valid_label = torch.tensor(valid_label)

train_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.TensorDataset(train_x, label),
                                           batch_size = batch_size, 
                                           shuffle = True)
valid_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.TensorDataset(valid_x, valid_label),
                                           batch_size = batch_size,
                                           shuffle = True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,20,5,1)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.conv2 = torch.nn.Conv2d(20,50,5,1)
        self.fc1 = torch.nn.Linear(5*5*50, 512)
        self.fc2 = torch.nn.Linear(512, 36)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x,2,2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x,2,2)
        x = x.view(-1, 5*5*50)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)


for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    total_loss_valid = 0
    total_acc_valid = 0
    
    for batch_idx, (x, target) in enumerate(train_loader):
        x = x.reshape(batch_size,1,32,32)
        out =  model(x)
        loss = criterion(out,target)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        _, predicted = torch.max(out.data, 1)

        total_acc += ((target==predicted).sum().item())
        total_loss += loss
    total_acc = total_acc/n_examples
    
    for batch_idx, (valid_x, valid_target) in enumerate(valid_loader):
        valid_x = valid_x.reshape(batch_size,1,32,32)
        valid_out =  model(valid_x)
        valid_loss = criterion(valid_out,valid_target)
        _, valid_predicted = torch.max(valid_out.data, 1)
        total_acc_valid += ((valid_target==valid_predicted).sum().item())
        total_loss_valid += valid_loss
    acc_valid = total_acc_valid/n_valid_examples

    train_acc_data.append(total_acc)
    valid_acc_data.append(acc_valid)
    train_loss_data.append(total_loss)
    valid_loss_data.append(valid_loss)    
    
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
