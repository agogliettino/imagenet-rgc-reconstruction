import os
import scipy as sp
import numpy as np
import sys
import visionloader as vl
sys.path.insert(0, '/home/agogliet/gogliettino/projects/natural-scenes-reco/repos/imagenet-rgc-reco/')
import src.preprocess as pp
import src.config as cfg
import visionloader as vl

"""
Script to combine data runs. Gets the common cells that were mapped between
two natural scenes runs and gets a subset of the preprocessed data.
"""

# Set paths to the data set.
dataset = '2018-08-07-5'
ns_datarun1 = 'yass_data001/data001'
ns_datarun2 = 'yass_data002/data002'

# Get the combined cellids dicts.
fnamein = os.path.join('./tmp',dataset,ns_datarun1,'cellids_dict.npy')
cellids_dict1 = np.load(fnamein,allow_pickle=True).item()
fnamein = os.path.join('./tmp',dataset,ns_datarun2,'cellids_dict.npy')
cellids_dict2 = np.load(fnamein,allow_pickle=True).item()
ns_datapath1 = os.path.join(cfg.PARENT_ANALYSIS,dataset,ns_datarun1)
ns_vcd1 = vl.load_vision_data(ns_datapath1,os.path.basename(ns_datarun1),
                              include_ei=True,include_neurons=True)
ns_datapath2 = os.path.join(cfg.PARENT_ANALYSIS,dataset,ns_datarun2)
ns_vcd2 = vl.load_vision_data(ns_datapath2,os.path.basename(ns_datarun2),
                              include_ei=True,include_neurons=True)
combined_cellids_dict = pp.combine_dataruns(cellids_dict1,cellids_dict2,
                                            ns_vcd1,ns_vcd2)
            
dirout = os.path.join('./tmp',dataset,os.path.dirname(ns_datarun1) +'-'+
                      os.path.dirname(ns_datarun2))

if not os.path.isdir(dirout):
    os.makedirs(dirout)

fnameout = os.path.join(dirout,'cellids_dict.npy')
np.save(fnameout, combined_cellids_dict)

# Get the common indices.
ns_cellids1 = sorted(list(cellids_dict1['ns_to_wn'].keys()))
ns_cellids2 = sorted(list(cellids_dict2['ns_to_wn'].keys()))
inds1 = []
inds2 = []

# Loop through the subset of mapped ns cells and get the original indices.
ns_cellids1_sub = np.asarray(sorted(list(
                                    combined_cellids_dict['ns1_to_ns2'].keys())))

for ns_cell1 in ns_cellids1_sub:
    inds1.append(ns_cellids1.index(ns_cell1))
    ns_cell2 = combined_cellids_dict['ns1_to_ns2'][ns_cell1]
    inds2.append(ns_cellids2.index(ns_cell2))

inds1 = np.asarray(inds1)
inds2 = np.asarray(inds2)

# Load in the training spike counts.
fnamein = os.path.join('./tmp',dataset,ns_datarun1,
                        'train_X.npy')
train_X1 = np.load(fnamein)
fnamein = os.path.join('./tmp',dataset,ns_datarun2,
                        'train_X.npy')
train_X2 = np.load(fnamein)

fnamein = os.path.join('./tmp',dataset,ns_datarun1,
                        'test_X.npy')
test_X1 = np.load(fnamein)
fnamein = os.path.join('./tmp',dataset,ns_datarun2,
                        'test_X.npy')
test_X2 = np.load(fnamein)

# Index accordingly into the response data.
train_X1 = train_X1[:,inds1]
train_X2 = train_X2[:,inds2]
test_X1 = test_X1[:,inds1]
test_X2 = test_X2[:,inds2]

# Get the natural scenes tensors.
fnamein = os.path.join('./tmp',dataset,ns_datarun1,
                        'train_Y.npy')
train_Y1 = np.load(fnamein)
fnamein = os.path.join('./tmp',dataset,ns_datarun2,
                        'train_Y.npy')
train_Y2 = np.load(fnamein)

fnamein = os.path.join('./tmp',dataset,ns_datarun1,
                        'test_Y.npy')
test_Y1 = np.load(fnamein)
fnamein = os.path.join('./tmp',dataset,ns_datarun2,
                        'test_Y.npy')
test_Y2 = np.load(fnamein)

# Write everything to disk.
fnameout = os.path.join(dirout,'train_X1.npy')
np.save(fnameout,train_X1)
fnameout = os.path.join(dirout,'train_X2.npy')
np.save(fnameout,train_X2)
fnameout = os.path.join(dirout,'test_X1.npy')
np.save(fnameout,test_X1)
fnameout = os.path.join(dirout,'test_X2.npy')
np.save(fnameout,test_X2)
fnameout = os.path.join(dirout,'train_Y1.npy')
np.save(fnameout,train_Y1)
fnameout = os.path.join(dirout,'train_Y2.npy')
np.save(fnameout,train_Y2)
fnameout = os.path.join(dirout,'test_Y1.npy')
np.save(fnameout,test_Y1)
fnameout = os.path.join(dirout,'test_Y2.npy')
np.save(fnameout,test_Y2)

# Combine everything into a single tensor for the end to end model.
dirout = os.path.join(dirout,'end-to-end')

if not os.path.isdir(dirout):
    os.makedirs(dirout)

train_X = np.r_[train_X1,train_X2]
train_Y = np.r_[train_Y1,train_Y2]

# Test data are the same file, just choose the first one.
test_X = test_X1
test_Y = test_Y1

fnameout = os.path.join(dirout,'train_X.npy')
np.save(fnameout,train_X)
fnameout = os.path.join(dirout,'train_Y.npy')
np.save(fnameout,train_Y)
fnameout = os.path.join(dirout,'test_X.npy')
np.save(fnameout,test_X)
fnameout = os.path.join(dirout,'test_Y.npy')
np.save(fnameout,test_Y)