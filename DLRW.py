
# coding: utf-8

# In[11]:

from __future__ import print_function
import sys
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as data_utils
from scipy.stats import mode
import time

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
nb_conv = 4

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader_all = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                      # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                     #  transforms.Normalize((0.1307,), (0.3081,))
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

#    train_data_all = train_data_all[0:10000,:,:]
    #train_target_all = train_target_all[0:10000]

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
    train_data = np.concatenate(train_data, axis = 0).astype("float32")
    train_target = np.concatenate(train_target, axis=0)
    return torch.from_numpy(train_data/255).float(), torch.from_numpy(train_target) , train_data_val/255,train_target_val, train_data_pool/255, train_target_pool

train_data, train_target, val_data, val_target, pool_data, pool_target = prepare_data()

train_loader = None
val_loader = None

def initialize_train_set():
    # Training Data set
    global train_loader
    global train_data 
    train = data_utils.TensorDataset(train_data, train_target)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

def initialize_val_set():
    global val_loader
    global val_data
    #Validation Dataset

    val = data_utils.TensorDataset(val_data,val_target)
    val_loader = data_utils.DataLoader(val,batch_size=batch_size, shuffle = True)


initialize_train_set()
initialize_val_set()



# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
# #         self.conv1 = nn.Conv2d(1, nb_filters, kernel_size=nb_conv)
# #         self.conv2 = nn.Conv2d(nb_filters, 20, kernel_size=5)
# #         self.conv2_drop = nn.Dropout2d()
# #         self.fc1 = nn.Linear(320, 50)
# #         self.fc2 = nn.Linear(50, 10)
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, nb_filters, kernel_size=nb_conv),
#             nn.ReLU(),
#             nn.Conv2d(nb_filters, nb_filters, kernel_size=nb_conv),
#             nn.ReLU(),
#             nn.MaxPool2d(nb_pool),
#             nn.Dropout(0.25),
#             nn.Conv2d(nb_filters, nb_filters*2, kernel_size=nb_conv),
#             nn.ReLU(),
#             nn.Conv2d(nb_filters*2, nb_filters*2, kernel_size=nb_conv),
#             nn.ReLU(),
#             nn.MaxPool2d(nb_pool),
#             nn.Dropout(0.25)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(1024,100),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(100,nb_classes)
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(batch_size, -1)
#         x = self.fc(x)
#         return F.Softmax(x)


class Net_Correct(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(Net_Correct, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, nb_filters, kernel_size=nb_conv),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.Conv2d(nb_filters, nb_filters, kernel_size=nb_conv),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.MaxPool2d(nb_pool),
            nn.Dropout2d(0.25),
            #nn.Conv2d(nb_filters, nb_filters*2, kernel_size=nb_conv),
            #nn.ReLU(),
            #nn.Conv2d(nb_filters*2, nb_filters*2, kernel_size=nb_conv),
            #nn.ReLU(),
            #nn.MaxPool2d(nb_pool),
            #nn.Dropout(0.25)

            
        )
        input_size = self._get_conv_output_size(input_shape)
        self.dense = nn.Sequential(nn.Linear(input_size,256))
	self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,nb_classes)
        )
    def _get_conv_output_size(self, shape):
        bs = batch_size
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(self.dense(x))
        return x

model = None
optimizer = None
def train(epoch):
    model.train()
    loss = None
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
    if epoch or  epochs:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))

    return loss.data[0]




def evaluate( input_data, stochastic = False, predict_classes=False):

    if stochastic:
        model.train() # we use dropout at test time
    else:
        model.eval()

    predictions = []
    test_loss = 0
    correct = 0
    for data, target in input_data:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        softmaxed = F.softmax(output.cpu())

        if predict_classes:
            predictions.extend(np.argmax(softmaxed.data.numpy(),axis = -1))
        else:
            predictions.extend(softmaxed.data.numpy())
        criterion = nn.CrossEntropyLoss()

        loss = criterion(output, target)

        test_loss += loss.data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    return (test_loss, correct, predictions)

def val(epoch):

    test_loss = 0
    correct = 0
    test_loss, correct,_ =  evaluate(val_loader, stochastic= False)

    test_loss /= len(val_loader) # loss function already averages over batch size


    if epoch == epochs:
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    return  test_loss, 100. * correct / len(val_loader.dataset)

def test(epoch):

    test_loss = 0
    correct = 0
    test_loss, correct,_ =  evaluate(test_loader, stochastic= False)

    test_loss /= len(test_loader) # loss function already averages over batch size
    if epoch or  epochs:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)

def getAcquisitionFunction(name):
    if name == "BALD":
        return bald_acquisition
    elif name == "VAR_RATIOS":
        return variation_ratios_acquisition
    elif name == "MAX_ENTROPY":
        return max_entroy_acquisition
    elif name == "MEAN_STD":
        return mean_std_acquisition
    else:
        print ("ACQUSITION FUNCTION NOT IMPLEMENTED")
        sys.exit(-1)


def acquire_points(argument,random_sample=False):
    global train_data
    global train_target
    
    acquisition_iterations = 98
    dropout_iterations = 50
    Queries = 10
    pool_all = np.zeros(shape=(1))

    if argument == "RANDOM":
        random_sample = True
    else :
        acquisition_function = getAcquisitionFunction(argument)

    val_loss_hist = []
    val_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    train_loss_hist = []
    for i in range(acquisition_iterations):
        pool_subset = 2000
        if random_sample:
            pool_subset = Queries
        print ("Acquisition Iteration " + str(i))
        pool_subset_dropout = torch.from_numpy( np.asarray(random.sample(range(0, pool_data.size(0)), pool_subset)))
        pool_data_dropout = pool_data[pool_subset_dropout]
        pool_target_dropout = pool_target[pool_subset_dropout]
        if random_sample is True:
            pool_index = np.array(range(0,Queries))

        else:
            points_of_interest = acquisition_function(dropout_iterations, pool_data_dropout, pool_target_dropout)
            pool_index = points_of_interest.argsort()[-Queries:][::-1]

            # np.save(argument + "_Points_scores_all.npy", points_of_interest)
            # np.save(argument + "_Points_data_all.npy", pool_data_dropout.numpy() )
            # np.save(argument + "_Points_target_all.npy", pool_target_dropout.numpy())
            # np.save(argument+ "_Points_scores.npy",points_of_interest[pool_index.numpy()])
            # np.save(argument+ "_Points_targets.npy",pool_target_dropout[pool_index].numpy())
            # np.save (argument+ "_Points_data.npy",pool_target_dropout[pool_index].numpy())
            # exit()

        pool_index = torch.from_numpy(np.flip(pool_index, axis=0).copy())

        pool_all = np.append(pool_all, pool_index)

        pooled_data = pool_data_dropout[pool_index]
        pooled_target = pool_target_dropout[pool_index]
        train_data  = torch.cat((train_data, pooled_data),0)
        train_target = torch.cat((train_target,pooled_target), 0)

        #remove from pool set
        remove_pooled_points(pool_subset,pool_data_dropout,pool_target_dropout,pool_index)
        train_loss, val_loss, test_loss, val_accuracy, test_accuracy = train_test_val_loop(init_train_set=True,disable_test=False)

        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_accuracy)

        test_loss_hist.append(test_loss)

        train_loss_hist.append(train_loss)

        test_acc_hist.append(test_accuracy)

    np.save("./val_loss_" + argument +".npy",np.asarray(val_loss_hist))
    np.save("./val_acc_" + argument + ".npy", np.asarray(val_acc_hist))
    np.save("./train_loss_" + argument + ".npy", np.asarray(train_loss_hist))
    np.save("./test_loss_" + argument + ".npy", np.asarray(test_loss_hist))
    np.save("./test_acc_" + argument + ".npy", np.asarray(test_acc_hist))

        #it seems the author deleted the acquired points from the train set, I don't think that it would be useful to do
        # because the data is randomly everytime, the probability of selecting the same batch is very very low

def remove_pooled_points(pool_subset, pool_data_dropout, pool_target_dropout, pool_index):
    global pool_data
    global pool_target
    np_data = pool_data.numpy()
    np_target = pool_target.numpy()
    pool_data_dropout = pool_data_dropout.numpy()
    pool_target_dropout = pool_target_dropout.numpy()
    np_index =  pool_index.numpy()
    np.delete(np_data, pool_subset,axis =0)
    np.delete(np_target, pool_subset,axis =0)

    np.delete(pool_data_dropout,np_index,axis = 0)
    np.delete(pool_target_dropout,np_index, axis=0)

    np_data = np.concatenate((np_data,pool_data_dropout),axis =0)
    np_target = np.concatenate((np_target, pool_target_dropout),axis =0)

    pool_data = torch.from_numpy(np_data)
    pool_target = torch.from_numpy(np_target)





def max_entroy_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print("MAX ENTROPY FUNCTION")
    score_All = np.zeros(shape=(pool_data_dropout.size(0), nb_classes))

    # Validation Dataset
    pool = data_utils.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, predictions = evaluate(pool_loader, stochastic=True)

        predictions = np.array(predictions)
        #predictions = np.expand_dims(predictions, axis=1)
        score_All = score_All + predictions
    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    Avg_Pi = np.divide(score_All, dropout_iterations)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

    U_X = Entropy_Average_Pi

    points_of_interest = U_X.flatten()
    return  points_of_interest


def mean_std_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print("MEAN STD ACQUISITION FUNCTION")
    all_dropout_scores = np.zeros(shape=(pool_data_dropout.size(0), 1))
    # Validation Dataset
    pool = data_utils.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, scores = evaluate(pool_loader, stochastic=True)

        scores = np.array(scores)
        all_dropout_scores = np.append(all_dropout_scores, scores, axis=1)
    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    std_devs= np.zeros(shape = (pool_data_dropout.size(0),nb_classes))
    sigma = np.zeros(shape = (pool_data_dropout.size(0)))
    for t in range(pool_data_dropout.size(0)):
        for r in range( nb_classes ):
            L = np.array([0])
            for k in range(r + 1, all_dropout_scores.shape[1], 10 ):
                L = np.append(L, all_dropout_scores[t, k])

            L_std = np.std(L[1:])
            std_devs[t, r] = L_std
        E = std_devs[t, :]
        sigma[t] = sum(E)/nb_classes


    points_of_interest = sigma.flatten()
    return points_of_interest

def bald_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print ("BALD ACQUISITION FUNCTION")
    score_all = np.zeros(shape=(pool_data_dropout.size(0), nb_classes))
    all_entropy = np.zeros(shape=pool_data_dropout.size(0))

    # Validation Dataset
    pool = data_utils.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, scores = evaluate(pool_loader, stochastic=True)

        scores = np.array(scores)
        #predictions = np.expand_dims(predictions, axis=1)
        score_all = score_all + scores

        log_score = np.log2(scores)
        entropy = - np.multiply(scores, log_score)
        entropy_per_dropout = np.sum(entropy,axis =1)
        all_entropy = all_entropy + entropy_per_dropout


    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    avg_pi = np.divide(score_all, dropout_iterations)
    log_avg_pi = np.log2(avg_pi)
    entropy_avg_pi = - np.multiply(avg_pi, log_avg_pi)
    entropy_average_pi = np.sum(entropy_avg_pi, axis=1)

    g_x = entropy_average_pi
    average_entropy = np.divide(all_entropy,dropout_iterations)
    f_x = average_entropy

    u_x = g_x - f_x


    # THIS FINDS THE MINIMUM INDEX
    # a_1d = U_X.flatten()
    # x_pool_index = a_1d.argsort()[-Queries:]


    points_of_interest = u_x.flatten()
    return  points_of_interest

def variation_ratios_acquisition(dropout_iterations, pool_data_dropout, pool_target_dropout):
    print("VARIATIONAL RATIOS ACQUSITION FUNCTION")
    All_Dropout_Classes = np.zeros(shape=(pool_data_dropout.size(0), 1))
    # Validation Dataset
    pool = data_utils.TensorDataset(pool_data_dropout, pool_target_dropout)
    pool_loader = data_utils.DataLoader(pool, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    for d in range(dropout_iterations):
        _, _, predictions = evaluate(pool_loader, stochastic=True,predict_classes=True)

        predictions = np.array(predictions)
        predictions = np.expand_dims(predictions, axis=1)
        All_Dropout_Classes = np.append(All_Dropout_Classes, predictions, axis=1)
    print("Dropout Iterations took --- %s seconds ---" % (time.time() - start_time))
    # print (All_Dropout_Classes)
    Variation = np.zeros(shape=(pool_data_dropout.size(0)))
    for t in range(pool_data_dropout.size(0)):
        L = np.array([0])
        for d_iter in range(dropout_iterations):
            L = np.append(L, All_Dropout_Classes[t, d_iter + 1])
        Predicted_Class, Mode = mode(L[1:])
        v = np.array([1 - Mode / float(dropout_iterations)])
        Variation[t] = v
    points_of_interest = Variation.flatten()
    return points_of_interest


def init_model():
    global model
    global optimizer
    model = Net_Correct()

    if cuda:
        model.cuda()

    decay = 3.5 / train_data.size(0)
    optimizer = optim.Adam([ 
			{'params':model.conv.parameters()},
			{'params':model.fc.parameters()},
                
                {'params': model.dense.parameters(), 'weight_decay': decay}
            ], lr=lr)
def train_test_val_loop(init_train_set, disable_test = True):
    if init_train_set:
        initialize_train_set()
    init_model()

    train_loss = 0
    val_loss = 0
    val_accuracy = 0
    test_loss = -1
    test_accuracy = -1
    print("Training again")
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        val_loss, val_accuracy = val(epoch)

    if disable_test is False:
        test_loss, test_accuracy = test(epoch)
    return  train_loss,val_loss,test_loss,val_accuracy,test_accuracy


def main(argv):
    start_time = time.time()
    print (str(argv[0]))
    initialize_train_set()
    init_model()
    print ("Training without acquisition")
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        val_loss, accuracy = val(epoch)
    print ("acquring points")
    acquire_points(str(argv[0]))
    init_model()
    print ("Training again")
    train_test_val_loop(init_train_set=True,disable_test=False)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main(sys.argv[1:])
