"""
Dataset loader class for minibatch gradient descent.
"""
import numpy as np
import os
import torch
import config as cfg

class Dataset():

    def __init__(self,datapath,data_split,model_type):

        """
        Constructor

        Parameters:
            datapath: parent path to the training and testing data
            data_split: either "train" or "test"
            model_type: either "end_to_end" or "recon"
        
        Returns:
            None
        """

        # If the model is end to end, we are loading in spikes.
        assert model_type in ['end_to_end','recon'],"unknown model type"

        if model_type in ['recon']:
            self.X = torch.tensor(np.moveaxis(np.load(os.path.join(datapath,
                            data_split + '_X.npy')),-1,1)).to(torch.float32)
        else:
            self.X = torch.tensor(np.moveaxis(np.load(os.path.join(datapath,
                         data_split + '_X.npy')),-1,1)).to(torch.float32)
        
        self.Y = torch.tensor(np.moveaxis(np.load(os.path.join(datapath,
                         data_split + '_Y.npy')),-1,1)).to(torch.float32)

        # Crop the image to get rid of zeros and have a nice even number.
        self.Y = self.Y[...,32:288] / 255

        if model_type in ['recon']:
            self.X = self.X[...,32:288] / 255

        # If test set only get first 150 TODO FIXME.
        if data_split in ['test']:
            self.Y = self.Y[0:150,...]
            self.X = self.X[0:150,...]

    def __getitem__(self,index):
        """
        Gets instance of the data, returns x,y pair.

        Parameters:
            index: index ...
        
        Returns:
            x,y pair
        """
        x = self.X[index,...]
        y = self.Y[index,...]

        return x,y

    def __len__(self):
        """ Returns number of samples in partition """
        return self.X.shape[0]
