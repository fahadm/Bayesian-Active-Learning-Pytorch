
# coding: utf-8

# In[11]:

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


cuda = False
batch_size = 128
nb_classes = 10

lr = 0.001
momentum = 0.9
log_interval = 100
epochs = 50

nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

def prepare_data():

    train_data_all = train_loader_all.dataset.train_data
    train_target_all = train_loader_all.dataset.train_labels
    shuffler_idx = torch.randperm(train_target_all.size(0))
    train_data_all = train_data_all[shuffler_idx]
    train_target_all = train_target_all[shuffler_idx]


    train_data = []
    train_target = []
    train_data_val = train_data_all[10000:10100, :,:]
    train_target_val = train_target_all[10000:10100]
    train_data_pool = train_data_all[20000:60000, :,:]
    train_target_pool = train_target_all[20000:60000]

    train_data_all = train_data_all[0:10000,:,:]
    train_target_all = train_target_all[0:10000]

    train_data_val.unsqueeze_(1)
    train_data_pool.unsqueeze_(1)
    train_data_all.unsqueeze_(1)

    train_data_pool = train_data_pool.float()
    train_data_val = train_data_val.float()
    train_data_all = train_data_all.float()

    for i in range(0,10):
        arr = np.array(np.where(train_target_all.numpy()==i))
        idx = np.random.permutation(arr)
        data_i =  train_data_all.numpy()[ idx[0][0:2], :,:,: ] # pick the first 2 elements of the shuffled idx array
        target_i = train_target_all.numpy()[idx[0][0:2]]
        train_data.append(data_i)
        train_target.append(target_i)
    train_data = np.concatenate(train_data, axis = 0).astype('float32')
    train_target = np.concatenate(train_target, axis=0)
    return torch.from_numpy(train_data/255).float(), torch.from_numpy(train_target) , train_data_val/255,train_target_val, train_data_pool/255, train_target_pool
train_data, train_target, val_data, val_target, pool_data, pool_target = prepare_data()


def initialize_train_set():
    # Training Data set
    global train_loader
    train = data_utils.TensorDataset(train_data, train_target)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

def initialize_val_set():
    global val_loader
    #Validation Dataset
    val = data_utils.TensorDataset(val_data,val_target)
    val_loader = data_utils.DataLoader(val,batch_size=batch_size, shuffle = True)


initialize_train_set()
initialize_val_set()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, nb_filters, kernel_size=nb_conv)
#         self.conv2 = nn.Conv2d(nb_filters, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
        self.conv = nn.Sequential(
            nn.Conv2d(1, nb_filters, kernel_size=nb_conv),
            nn.ReLU(),
            nn.Conv2d(nb_filters, nb_filters, kernel_size=nb_conv),
            nn.ReLU(),
            nn.MaxPool2d(nb_pool),
            nn.Dropout(0.25),
            nn.Conv2d(nb_filters, nb_filters*2, kernel_size=nb_conv),
            nn.ReLU(),
            nn.Conv2d(nb_filters*2, nb_filters*2, kernel_size=nb_conv),
            nn.ReLU(),
            nn.MaxPool2d(nb_pool),
            nn.Dropout(0.25)
        )
        input_size = self._get_conv_output_size(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(input_size,100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100,nb_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return F.Softmax(x)
    def _get_conv_output_size(self, shape):
        bs = batch_size
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size






if cuda:
    model.cuda()

decay = 10/10000
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay= decay )

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)


# In[ ]:
