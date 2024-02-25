import os
import scipy as sp
import numpy as np
import sys
import visionloader as vl
import rawmovie as rm
import whitenoise.random_noise as rn
import src.config as cfg

"""
Miscellaneous io utilities.
"""

def get_stimulus_tensor(stimulus_path,stimulus_type,
                        n_unique_frames,grayscale=False):

    """
    Utility for getting stimulus tensors from disk, or generating on the fly
    if random noise.

    Parameters:
        stimulus_path: path to stimulus
        stimulus_type: either "naturalscenes" or "whitenoise"
        n_unique_frames: number of unique frames (int)
        grayscale: boolean indicating to keep all channels or not. 
                    Keeps singleton dim if grayscaling.

    Returns: 
        stimulus tensor of size n,y,x,c 
    """
    assert os.path.isfile(stimulus_path), "Stimulus provided not found."
    assert stimulus_type in ["whitenoise","naturalscenes"],\
                "stimulus_type must be 'whitenoise' or 'naturalscenes'."

   # Initialize the stimulus object.
    if stimulus_type in ['naturalscenes']:
        rm_obj = rm.RawMovieReader(stimulus_path)
        stimulus_tensor,_ = rm_obj.get_frame_sequence(0,n_unique_frames)

    else:
        assert False, 'other stimuli not implemented yet.'

    if grayscale:
        return np.mean(stimulus_tensor,axis=-1,keepdims=True)

    return stimulus_tensor
