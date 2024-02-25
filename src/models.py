import numpy as np
import torch.nn as nn
import pdb

"""
Contains various models to be used in the project.
"""

class CNN(nn.Module):

    """
    CNN to enhance linear reconstructed image
    """
    def __init__(self):
        super(CNN, self).__init__()

        """ Downsample """

        # Layer 1 
        self.conv0 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=1,out_channels=64,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )
        
        # Layer 2
        self.conv1 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=64,out_channels=128,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )

        # Layer 3
        self.conv2 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=128,out_channels=256,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )

        # Layer 4 
        self.conv3 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=256,out_channels=256,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )

        """ Upsample """

        # Layer 5
        self.conv4 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=256,out_channels=256,
                                        kernel_size=4,stride=2,padding=1),
         
                     nn.ReLU()
                     )

        # Layer 6 
        self.conv5 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=256,out_channels=128,
                                        kernel_size=4,stride=2,padding=1),
                     nn.ReLU()
                     )

        # Layer 7
        self.conv6 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=128,out_channels=64,
                                        kernel_size=4,stride=2,padding=1),
                     nn.ReLU()
                     )

        # Layer 8 
        self.conv7 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=64,out_channels=1,
                                        kernel_size=4,stride=2,padding=1),
                     #nn.ReLU()
                     )

    def forward(self,x):

        x1 = self.conv0(x)

        x2 = self.conv1(x1)

        x3 = self.conv2(x2)

        x4 = self.conv3(x3)

        x5 = self.conv4(x4)

        x6 = self.conv5(x5)

        x7 = self.conv6(x6)

        y = self.conv7(x7)

        return y

class CNNEndtoEnd(nn.Module):
    """
    CNN that end to end learns a mapping from response matrix to stimuli
    """
    def __init__(self):
        super(CNNEndtoEnd, self).__init__()

        # Fully connected layer
        self.fc1 = nn.Sequential(
                   nn.Linear(421,2560), # Hardcoded to work for the number of units
                   nn.ReLU()
                   )

        """ Upsample """

        self.conv0 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=1,out_channels=1,
                                        kernel_size=4,stride=2,padding=1),
         
                     nn.ReLU()
                     )

        self.conv1 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=1,out_channels=1,
                                        kernel_size=4,stride=2,padding=1),
                     nn.ReLU()
                     )

        self.conv2 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=1,out_channels=1,
                                        kernel_size=4,stride=2,padding=1),
                     nn.ReLU()
                     )

        self.conv3 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=1,out_channels=1,
                                        kernel_size=4,stride=2,padding=1),
                     nn.ReLU()
                     )
        
        """ Downsample """

        self.conv4 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=1,out_channels=64,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )
        
        self.conv5 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=64,out_channels=128,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )

        self.conv6 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=128,out_channels=256,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )

        self.conv7 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=256,out_channels=256,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )

        """ Upsample """

        self.conv8 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=256,out_channels=256,
                                        kernel_size=4,stride=2,padding=1),
         
                     nn.ReLU()
                     )

        self.conv9 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=256,out_channels=128,
                                        kernel_size=4,stride=2,padding=1),
                     nn.ReLU()
                     )

        self.conv10 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=128,out_channels=64,
                                        kernel_size=4,stride=2,padding=1),
                     nn.ReLU()
                     )

        self.conv11 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=64,out_channels=1,
                                        kernel_size=4,stride=2,padding=1),
                     #nn.ReLU()
                     )
        
    def forward(self,x):

        x1 = self.fc1(x)

        x1 = x1.view(x1.shape[0],1,40,64)

        #x1 = self.conv0(x1)

        #x2 = self.conv1(x1)

        x3 = self.conv2(x1)

        x4 = self.conv3(x3)

        # Downsample

        x5 = self.conv4(x4)

        x6 = self.conv5(x5)

        x7 = self.conv6(x6)
        
        x8 = self.conv7(x7)

        # Upsample
        x9 = self.conv8(x8)

        x10 = self.conv9(x9)

        x11 = self.conv10(x10)

        y = self.conv11(x11)

        return y

class CNNEndtoEndFC(nn.Module):
    """
    CNN that end to end learns a mapping from response matrix to stimuli
    """
    def __init__(self):
        super(CNNEndtoEndFC, self).__init__()

        # Fully connected layer
        self.fc1 = nn.Sequential(
                   #nn.Linear(421,163840), # Hardcoded to work for the number of units
                   nn.Linear(708,163840), # Hardcoded to work for the number of units
                   nn.ReLU()
                   )

        # Layer 1 
        self.conv0 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=4,out_channels=64,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )
        
        # Layer 2
        self.conv1 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=64,out_channels=128,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )

        # Layer 3
        self.conv2 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=128,out_channels=256,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )

        # Layer 4 
        self.conv3 = nn.Sequential(
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(in_channels=256,out_channels=256,
                               kernel_size=4,stride=2,padding=0),
                     nn.ReLU()
                     )

        """ Upsample """

        # Layer 5
        self.conv4 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=256,out_channels=256,
                                        kernel_size=4,stride=2,padding=1),
         
                     nn.ReLU()
                     )

        # Layer 6 
        self.conv5 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=256,out_channels=128,
                                        kernel_size=4,stride=2,padding=1),
                     nn.ReLU()
                     )

        # Layer 7
        self.conv6 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=128,out_channels=64,
                                        kernel_size=4,stride=2,padding=1),
                     nn.ReLU()
                     )

        # Layer 8 
        self.conv7 = nn.Sequential(
                     nn.ConvTranspose2d(in_channels=64,out_channels=1,
                                        kernel_size=4,stride=2,padding=1),
                     #nn.ReLU()
                     )

    def forward(self,x):

        x1 = self.fc1(x)

        x1 = x1.view(x1.shape[0],4,160,256)

        x1 = self.conv0(x1)

        x2 = self.conv1(x1)

        x3 = self.conv2(x2)

        x4 = self.conv3(x3)

        x5 = self.conv4(x4)

        x6 = self.conv5(x5)

        x7 = self.conv6(x6)

        y = self.conv7(x7)

        return y