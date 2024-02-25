import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import os
#import rawmovie as rmv
from scipy import linalg
import visionloader as vl
import torch
import torch.nn as nn
sys.path.insert(0, '/home/agogliet/gogliettino/projects/natural-scenes-reco/repos/imagenet-rgc-reco/')
from src.Dataset import Dataset
import src.models as models
import torch.optim as optim
import time

"""
Script to train the model on the linear reconstructed and ground truth data.
"""

# Initialize the training and test loaders.
datapath = './tmp/2018-08-07-5/yass_data001-yass_data002/linear-recons'
train_data = Dataset(datapath,data_split='train',model_type='recon')
test_data = Dataset(datapath,data_split='test',model_type='recon')
"""
datapath = './tmp/2018-08-07-5/yass_data001-yass_data002/end-to-end'
train_data = Dataset(datapath,data_split='train',model_type='end_to_end')
test_data = Dataset(datapath,data_split='test',model_type='end_to_end')
"""
BATCH_SIZE = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                             shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=150,
                                        shuffle=False)

# Set device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the loss function as MSE loss
criterion = nn.MSELoss()
Model = models.CNN() # for linear
#Model = models.CNNEndtoEndFC() # for end-to-end

# Define the optimizer with a learning rate
optimizer = optim.Adam(Model.parameters(), lr=.00005)
Model = Model.to(device)

# Train the model and cache the losses.
start_time = time.time()

# Number of training epochs
N_EPOCHS = 20 
cost_cache = dict()
cost_cache['train'] = []
cost_cache['test'] = []

# Get the test data.
X_test = test_loader.dataset.X.to(device)
Y_test = test_loader.dataset.Y.to(device)

for epoch in range(N_EPOCHS): 
    
      for i, (X, Y) in enumerate(train_loader,0):
            
        X = X.to(device)
        Y = Y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward propogation
        Y_hat = Model(X)

        # calculate the loss
        loss = criterion(Y_hat, Y)

        # backpropogation + update parameters
        loss.backward()
        optimizer.step()

        # print statistics
        cost = loss.item()
        cost_cache['train'].append(cost)
        
        # Also get test cost.
        Y_hat_test = Model(X_test)
        loss = criterion(Y_hat_test,Y_test)
        test_cost = loss.item()
        cost_cache['test'].append(cost)

        if i % 100 == 0:
          print('Epoch: ' + str(epoch) + ", Iteration: " + str(i) 
                  + ", training cost = " + str(cost),
                  ", testing cost = " + str(test_cost))
        
print("--- %s seconds ---" % (time.time() - start_time))

# Make a dictionary of everything and write to disk.
model_dict = dict()
model_dict['trained_model'] = Model
model_dict['loss'] = loss
model_dict['cost_cache'] = cost_cache
model_dict['X_test'] = X_test.detach().cpu().numpy()
model_dict['Y_hat_test'] = Y_hat_test.detach().cpu().numpy()
model_dict['Y_test'] = Y_test.detach().cpu().numpy()

dirout = os.path.join(datapath,'trained-model')

if not os.path.isdir(dirout):
    os.makedirs(dirout)

fnameout = f'trained_model_{N_EPOCHS}_epochs.npy'
fnameout = os.path.join(dirout,fnameout)
np.save(fnameout,model_dict)