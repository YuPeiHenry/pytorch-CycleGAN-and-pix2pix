import os.path
from data.base_dataset import BaseDataset
from data.exr_dataset import ExrDataset
from data.image_folder import make_dataset
import numpy as np
import torch
import data.exrlib as exrlib

class ExrFlowmapDataset(ExrDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        super().__init__(opt)

        self.Flowmap = os.path.join(opt.dataroot, opt.phase + '_flowmap')

        self.Flowmap_paths = sorted(make_dataset(self.Flowmap, opt.max_dataset_size))
        self.Flowmap_size = len(self.Flowmap_paths)  # get the size of dataset B

        self.Flowmap_test_paths = sorted(make_dataset(os.path.join(opt.dataroot, 'test_flowmap')))
        self.Flowmap_test_size = len(self.Flowmap_test_paths)

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
        dict = super().__getitem__(index)
        Flowmap_path = self.Flowmap_paths[index % self.Flowmap_size]
        Flowmap_img = exrlib.read_exr_float32(Flowmap_path, list(['0']), 512, 512)
        dict["Flowmap"] = (torch.Tensor(np.transpose(Flowmap_img, (2, 0, 1))) - 0.5) / 0.5

        return dict

    def get_val_item(self, index):
        dict = super().get_val_item(index)
        Flowmap_path = self.Flowmap_test_paths[index % self.Flowmap_size]
        Flowmap_img = exrlib.read_exr_float32(Flowmap_path, list(['0']), 512, 512)
        dict["Flowmap"] = (torch.Tensor(np.transpose(Flowmap_img, (2, 0, 1))) - 0.5) / 0.5

        return dict