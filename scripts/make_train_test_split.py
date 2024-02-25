import os
import scipy as sp
import numpy as np
import sys
import visionloader as vl
sys.path.insert(0, '/home/agogliet/gogliettino/projects/natural-scenes-reco/repos/imagenet-rgc-reco/')
import src.preprocess as pp
import src.io_util as io
import src.config as cfg
import visionloader as vl
import rawmovie as rmv

"""
Script to split the data into training and testing partitions.
"""
N_TRAIN = 10000
N_TEST = 150
N_BLOCKS = 10
N_MS_POST_FLASH = 150

# Set paths to the data set.
dataset = '2018-08-07-5'
wn_datarun = 'yass_data000/data000'
ns_datarun = 'yass_data002/data002'
wn_datapath = os.path.join(cfg.PARENT_ANALYSIS,dataset,wn_datarun)
ns_datapath = os.path.join(cfg.PARENT_ANALYSIS,dataset,ns_datarun)

# Get the training and testing stimuli.
train_stimulus_path = '/Volumes/Data/Stimuli/movies/imagenet/ImageNet_stix2_1_045.rawMovie'
test_stimulus_path = '/Volumes/Data/Stimuli/movies/imagenet/ImageNetTest_v2.rawMovie'

# Load in spike trains and compute the training/testing indices.
fnamein = os.path.join('tmp',dataset,ns_datarun,'binned_spikes_tensor.npy')
binned_spikes_tensor = np.load(fnamein)

train_inds = []
i = 0
cnt = 0

while i < binned_spikes_tensor.shape[0] and cnt < N_TRAIN:

    if cnt != 0 and cnt % int(N_TRAIN / N_BLOCKS) == 0:
        i += N_TEST

    train_inds.append(i)
    i+=1
    cnt +=1

train_inds = np.asarray(train_inds)
test_inds = np.sort(np.setdiff1d(np.arange(0,binned_spikes_tensor.shape[0]),
                        train_inds))

# Get the training and testing stimuli.
train_ns_tensor = io.get_stimulus_tensor(train_stimulus_path,'naturalscenes',
                                    N_TRAIN,grayscale=True)
test_ns_tensor = io.get_stimulus_tensor(test_stimulus_path,'naturalscenes',
                                    N_TEST,grayscale=True)

# Crop the tensors (zero padded at the edges) and scale 0-1.
"""
train_ns_tensor = train_ns_tensor[...,33:288,:]
train_ns_tensor /= 255.0
test_ns_tensor = test_ns_tensor[...,33:288,:]
test_ns_tensor /= 255.0
"""

# For flashed stimuli, convert to spike counts.
spike_counts_tensor = np.sum(binned_spikes_tensor[...,0:N_MS_POST_FLASH],
                             axis=-1)

# Write the training and testing partitions to disk.
train_X = spike_counts_tensor[train_inds,...]
test_X = spike_counts_tensor[test_inds,...]
train_Y = train_ns_tensor
test_Y = test_ns_tensor

fnameout = os.path.join('tmp',dataset,ns_datarun,'train_X.npy')
np.save(fnameout,train_X)

fnameout = os.path.join('tmp',dataset,ns_datarun,'test_X.npy')
np.save(fnameout,test_X)

fnameout = os.path.join('tmp',dataset,ns_datarun,'train_Y.npy')
np.save(fnameout,train_Y)

fnameout = os.path.join('tmp',dataset,ns_datarun,'test_Y.npy')
np.save(fnameout,test_Y)