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

"""
Script to preprocess natural scenes data.
"""

# Set paths to the data set.
dataset = '2018-08-07-5'
wn_datarun = 'yass_data000/data000'
ns_datarun = 'yass_data002/data002'
wn_datapath = os.path.join(cfg.PARENT_ANALYSIS,dataset,wn_datarun)
ns_datapath = os.path.join(cfg.PARENT_ANALYSIS,dataset,ns_datarun)

# From the white noise data, get the celltypes dictionary.
wn_vcd = vl.load_vision_data(wn_datapath,os.path.basename(wn_datarun),
                            include_neurons=True,include_ei=True,
                            include_noise=True)
wn_cellids = sorted(wn_vcd.get_cell_ids())

f = open(os.path.join(wn_datapath,
         cfg.CLASS_FNAME%os.path.basename(wn_datarun)))
celltypes_dict = dict()

for j in f:
    tmp = ""

    for jj,substr in enumerate(j.split()[1:]):
        tmp +=substr

        if jj < len(j.split()[1:])-1:
            tmp += " "

    celltypes_dict[int(j.split()[0])] = tmp

f.close()

# Map the cellids between runs and write dictionary to disk.
ns_vcd = vl.load_vision_data(ns_datapath,os.path.basename(ns_datarun),
                                include_neurons=True,include_ei=True,
                                include_noise=True)
ns_cellids = sorted(ns_vcd.get_cell_ids())
print('mapping cell IDs ... ')
cellids_dict = pp.map_wn_to_ns_cellids(ns_vcd,ns_cellids,wn_vcd,
                                        wn_cellids,celltypes_dict)
print('done')

dirout = './tmp/%s/%s'%(dataset,ns_datarun)

if not os.path.isdir(dirout):
    os.makedirs(dirout)

fnameout = os.path.join(dirout,"cellids_dict.npy")
np.save(fnameout,cellids_dict,allow_pickle=True)

# Get the frame times: 120 Hz monitor with a trigger set at 500ms means 60.
frames_per_ttl = 60
frame_times = pp.get_frame_times(ns_vcd,frames_per_ttl)

# Get the spike counts.
ns_cellids_of_int = sorted(list(cellids_dict['ns_to_wn'].keys()))
binned_spikes_tensor = pp.get_binned_spikes(ns_vcd,
                                            ns_cellids_of_int,
                                            frame_times)

# Write this to disk.
dirout = './tmp/%s/%s'%(dataset,ns_datarun)

if not os.path.isdir(dirout):
    os.makedirs(dirout)

fnameout = os.path.join(dirout,"binned_spikes_tensor.npy")
np.save(fnameout,binned_spikes_tensor,allow_pickle=True)