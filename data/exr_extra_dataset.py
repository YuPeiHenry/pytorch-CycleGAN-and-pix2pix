import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import numpy as np
import torch
import data.exrlib as exrlib

class ExrExtraDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        super().__init__(self, opt)

        self.Extra = os.path.join(opt.dataroot, opt.phase + '_extra')

        self.Extra_paths = sorted(make_dataset(self.Extra, opt.max_dataset_size))
        self.Extra_size = len(self.Extra_paths)  # get the size of dataset B

        self.Extra_test_paths = sorted(make_dataset(os.path.join(opt.dataroot, 'test_extra')))
        self.Extra_test_size = len(self.Extra_test_paths)

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
        dict = super().__getitem__(self, index)
        Extra_path = self.Extra_paths[index % self.Extra_size]
        Extra_img = exrlib.read_exr_float32(B_path, list(['0', '1']), 512, 512)
        dict["Extra"] = Extra

        return dict

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A1_size, self.B_size)

    def get_val_item(self, index):
        dict = super().get_val_item(self, index)
        Extra_path = self.Extra_test_paths[index % self.Extra_size]
        Extra_img = exrlib.read_exr_float32(B_path, list(['0', '1']), 512, 512)
        dict["Extra"] = Extra

        return dict