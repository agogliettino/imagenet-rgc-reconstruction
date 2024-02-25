"""
Script to learn linear filters using least squares regression on half of the 
training data and write out reconstructed images.
"""
import os
import scipy as sp
import numpy as np
import sys
import visionloader as vl
sys.path.insert(0, '/home/agogliet/gogliettino/projects/natural-scenes-reco/repos/imagenet-rgc-reco/')
import src.preprocess as pp
import src.config as cfg
import visionloader as vl
import pdb
from scipy import linalg
import shutil

# Set paths.
dataset = '2018-08-07-5'
ns_datarun1 = 'yass_data001/data001'
ns_datarun2 = 'yass_data002/data002'
dirin = os.path.join('./tmp',dataset,os.path.dirname(ns_datarun1) +'-'+
                      os.path.dirname(ns_datarun2))

# Learn weights on the first half of the data.
fnamein = os.path.join(dirin,'train_X1.npy')
train_X1 = np.load(fnamein)

fnamein = os.path.join(dirin,'train_Y1.npy')
train_Y1 = np.load(fnamein)

n,y,x = np.squeeze(train_Y1).shape
train_Y1 = np.reshape(train_Y1,(n,y * x))
W1 = linalg.inv(train_X1.T@train_X1)@train_X1.T@train_Y1
fnameout = os.path.join(dirin,'W1.npy')
np.save(fnameout,W1)

# Get the next set of training responses and apply the weights to get recons.
fnamein = os.path.join(dirin,'train_X2.npy')
train_X2 = np.load(fnamein)
train_Y2_hat = train_X2@W1

# Reshape into image.
n = train_Y2_hat.shape[0]
train_Y2_hat = np.reshape(train_Y2_hat,(n,y,x))[...,None]
fnameout = os.path.join(dirin,'train_Y2_hat.npy')
np.save(fnameout,train_Y2_hat)

# Do the same for the testing data.
fnamein = os.path.join(dirin,'test_X2.npy')
test_X2 = np.load(fnamein)
test_Y2_hat = test_X2@W1

# Reshape into image.
n = test_Y2_hat.shape[0]
test_Y2_hat = np.reshape(test_Y2_hat,(n,y,x))[...,None]
fnameout = os.path.join(dirin,'test_Y2_hat.npy')
np.save(fnameout,test_Y2_hat)

# Move all this to a separate directory for the CNN model.
dirout = os.path.join(dirin,'linear-recons')

if not os.path.isdir(dirout):
    os.makedirs(dirout)

# For CNN, the recons will be inputs (X) and the GT will be outputs (Y).
fnamein = os.path.join(dirin,'train_Y2_hat.npy')
fnameout = os.path.join(dirout,'train_X.npy')
shutil.copy(fnamein, fnameout)

fnamein = os.path.join(dirin,'test_Y2_hat.npy')
fnameout = os.path.join(dirout,'test_X.npy')
shutil.copy(fnamein, fnameout)

fnamein = os.path.join(dirin,'train_Y2.npy')
fnameout = os.path.join(dirout,'train_Y.npy')
shutil.copy(fnamein, fnameout)

fnamein = os.path.join(dirin,'test_Y2.npy')
fnameout = os.path.join(dirout,'test_Y.npy')
shutil.copy(fnamein, fnameout)
