import os
import scipy as sp
import numpy as np
import sys
import visionloader as vl
import rawmovie as rm
import whitenoise.random_noise as rn
import src.config as cfg

""" 
Utility to bin spike trains and get stimulus tensors and ultimately write
a training and testing partition to disk.
"""

def map_wn_to_ns_cellids(ns_vcd,ns_cellids,wn_vcd,
                         wn_cellids,celltypes_dict,
                         corr_dict = {'on parasol': .95,'off parasol': .95,
                                       'on midget': .95,'off midget': .95},
                         mask=False,n_sigmas=None):
    """
    Maps WN to NS EIs according to a threshold value of correlation. Computes
    EI power over space, both with and without masking (user choice). Does a pass
    over the NS cellids and finds the corresponding WN cell. If none is found, 
    the cell doesn't get mapped (does not appear in the dictionary).

    Parameters:
        ns_vcd: natural scenes vision data object
        ns_cellids: natural scenes cellids to map
        wn_cellids: white noise cellids to map
        celltypes_dict: dictionary mapping white noise cell ids to celltype.
    """

    # Get elec coords in case plot and noise if needed.
    coords = wn_vcd.electrode_map
    xcoords = coords[:,0]
    ycoords = coords[:,1]
    channel_noise = wn_vcd.channel_noise

    # Initialize a dictionary and loop over the cells.
    cellids_dict = dict()

    for key in ['wn_to_ns','ns_to_wn','celltypes']:
        cellids_dict[key] = dict()

    for wn_cell in wn_cellids:

        # Get the cell type and write as well.
        celltype = celltypes_dict[wn_cell].lower()

        # Hardcode these for now TODO FIXME
        if "on" in celltype and "parasol" in celltype:
            celltype = 'on parasol'
        elif "off" in celltype and "parasol" in celltype:
            celltype = "off parasol"
        elif "on" in celltype and "midget" in celltype:
            celltype = 'on midget'
        elif "off" in celltype and "midget" in celltype:
            celltype = 'off midget'
        else:
            continue

        # If masking, only look at the significant indices. 
        wn_cell_ei = wn_vcd.get_ei_for_cell(wn_cell).ei

        if mask and n_sigmas is not None:
            sig_inds = np.argwhere(np.abs(np.amin(wn_cell_ei,axis=1))
                                   > n_sigmas * channel_noise).flatten()
            wn_cell_ei_power = np.zeros(wn_cell_ei.shape[0])
            wn_cell_ei_power[sig_inds] = np.sum(wn_cell_ei[sig_inds,:]**2,
                                                axis=1)
        else:
            wn_cell_ei_power = np.sum(wn_cell_ei**2,axis=1)

        corrs = []

        for ns_cell in ns_cellids:
            ns_cell_ei = ns_vcd.get_ei_for_cell(ns_cell).ei

            if mask and n_sigmas is not None:
                sig_inds = np.argwhere(np.abs(np.amin(ns_cell_ei,axis=1))
                                       > n_sigmas * channel_noise).flatten()
                ns_cell_ei_power = np.zeros(ns_cell_ei.shape[0])
                ns_cell_ei_power[sig_inds] = np.sum(ns_cell_ei[sig_inds,:]**2,
                                                axis=1)
            else:
                ns_cell_ei_power = np.sum(ns_cell_ei**2,axis=1)

            corr = np.corrcoef(wn_cell_ei_power,ns_cell_ei_power)[0,1]
            corrs.append(corr)

        # Take the cell with the largest correlation.
        if np.max(corrs) < corr_dict[celltype]:
            continue

        max_ind = np.argmax(np.asarray(corrs))
        cellids_dict['wn_to_ns'][wn_cell] = ns_cellids[max_ind]
        cellids_dict['ns_to_wn'][ns_cellids[max_ind]] = wn_cell
        cellids_dict['celltypes'][wn_cell] = celltype

        # Once the cell has been mapped, remove it (hack) FIXME.
        ns_cellids.remove(ns_cellids[max_ind]) 

    return cellids_dict

def combine_dataruns(cellids_dict1,cellids_dict2,vcd1,vcd2,min_corr=.95):

    # Get mapped IDs. 
    mapped_ns_cellids1 = sorted(list(cellids_dict1['ns_to_wn'].keys()))
    mapped_ns_cellids2 = sorted(list(cellids_dict2['ns_to_wn'].keys()))
    
    # Loop through the first set of cells and greedily match.
    combined_cellids_dict = dict()
    combined_cellids_dict['wn_to_ns1'] = dict()
    combined_cellids_dict['wn_to_ns2'] = dict()
    combined_cellids_dict['ns1_to_ns2'] = dict()
    combined_cellids_dict['ns2_to_ns1'] = dict()
    combined_cellids_dict['celltypes'] = dict()

    for ns_cell1 in mapped_ns_cellids1:
        ns_cell1_ei = vcd1.get_ei_for_cell(ns_cell1).ei
        ns_cell1_ei_power = np.sum(ns_cell1_ei**2,axis=1)

        corrs = []

        for ns_cell2 in mapped_ns_cellids2:
            ns_cell2_ei = vcd2.get_ei_for_cell(ns_cell2).ei
            ns_cell2_ei_power = np.sum(ns_cell2_ei**2,axis=1)
            corr = np.corrcoef(ns_cell1_ei_power,ns_cell2_ei_power)[0,1]
            corrs.append(corr)

        if np.max(corrs) < min_corr:
            continue
        
        # Greedily map the two cells.
        max_ind = np.argmax(np.asarray(corrs))
        ns_cell2 = mapped_ns_cellids2[max_ind]
        combined_cellids_dict['ns1_to_ns2'][ns_cell1] = ns_cell2
        combined_cellids_dict['ns2_to_ns1'][ns_cell2] = ns_cell1

        # For this part, we just take the label of the seed cell.
        wn_cell = cellids_dict1['ns_to_wn'][ns_cell1]
        celltype = cellids_dict1['celltypes'][wn_cell]
        combined_cellids_dict['wn_to_ns1'][wn_cell] = ns_cell1
        combined_cellids_dict['wn_to_ns2'][wn_cell] = ns_cell2
        combined_cellids_dict['celltypes'][wn_cell] = celltype

        # Once the cell has been mapped, remove it (hack) FIXME.
        mapped_ns_cellids2.remove(ns_cell2)

    return combined_cellids_dict

def get_frame_times(vcd,frames_per_ttl):
    """
    Gets times of each frame of the stimulus by linear interpolating the ttl 
    times

    Parameters:
        vcd: vision data table object
        frames_per_ttl: the number of stimulus frames (set) to be between ttls

    Returns:
        vector of the approximate frame times.
    """
    ttl_times = (vcd.ttl_times.astype(float) / cfg.FS) * 1000 # Convert to ms.
    frame_times = []

    for i in range(ttl_times.shape[0]-1):
        frame_times.append(np.linspace(ttl_times[i],
                               ttl_times[i+1],
                               frames_per_ttl,
                               endpoint=False))
    
    return np.asarray(frame_times)

def get_binned_spikes(vcd,cells,frame_times):
    """
    Bins spike trains with 1 ms precision. Uses the time of the first and last
    frame as a reference, since the in-between frames are linearly interpolated
    from ttl times anyway.

    Parameters
        vcd: vision data object
        cells: cells of interest
        frame_times: matrix of size stimulus by frame number
    """
    
    # Get the bin edges based on the length of the stimulus. 
    n_ms = int(frame_times.shape[1] / cfg.MONITOR_FS * 1000)
    bin_edges = np.linspace(0,n_ms,n_ms+1)
    binned_spikes_tensor = []

    for i in range(frame_times.shape[0]):

        if i % 500 == 0:
            print(i / frame_times.shape[0])

        tmp = []
        t0 = frame_times[i,0]
        t_end = frame_times[i,-1]

        for cell in cells:
            spike_times = (vcd.get_spike_times_for_cell(cell) / cfg.FS) * 1000

            # Prune spike times that are within the interval of stimulus.
            spike_times = spike_times[np.where((spike_times > t0) & 
                                               (spike_times <= t_end))[0]]
            binned_spike_times = np.histogram(spike_times-t0,bin_edges)[0]
            tmp.append(binned_spike_times)

        tmp = np.asarray(tmp)
        binned_spikes_tensor.append(tmp)

    return np.asarray(binned_spikes_tensor)