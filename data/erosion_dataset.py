import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import cv2
import numpy as np
import torch
import random


class ErosionDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        #self.A = os.path.join(opt.dataroot, opt.phase + '_input')
        self.A1 = os.path.join(opt.dataroot, opt.phase + '_input_terraform')
        self.B = os.path.join(opt.dataroot, opt.phase + '_output')

        #self.A_paths = sorted(make_dataset(self.A, opt.max_dataset_size))
        self.A1_paths = sorted(make_dataset(self.A1, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.B, opt.max_dataset_size))
        #self.A_size = len(self.A_paths)  # get the size of dataset A
        self.A1_size = len(self.A1_paths)
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        #A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A1_path = self.A1_paths[index % self.A1_size]
        B_path = self.B_paths[index % self.B_size]
        #A_img = cv2.imread(A_path, -1)
        A1_img = cv2.imread(A1_path, -1)
        B_img = cv2.imread(B_path, -1)
        #A = torch.Tensor(A_img)
        A1 = self.convert(A1_img)
        B = self.convert(np.expand_dims(B_img, axis=2))

        return {'A': A1, 'B': B, 'A_paths': A1_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A1_size, self.B_size)

    def convert(self, image):
        return torch.Tensor((np.transpose(image, (2, 0, 1)) - self.opt.image_value_bound / 2) / (self.opt.image_value_bound / 2))
