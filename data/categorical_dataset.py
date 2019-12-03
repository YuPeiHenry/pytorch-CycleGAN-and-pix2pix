import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import csv
import torch
import numpy as np

class CategoricalDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_image = os.path.join(opt.dataroot, "full_images")  # create a path '/path/to/data/train'
        self.dir_class = os.path.join(opt.dataroot, opt.class_csv)

        self.paths = sorted(make_dataset(self.dir_image, opt.max_dataset_size))
        self.size = len(self.paths)  # get the size of dataset A
        input_nc = self.opt.input_nc
        self.transform = get_transform(self.opt, grayscale=(input_nc == 1))

        self.current_classes = 0
        class_to_int = {}

        self.class_dict = {}

        with open(self.dir_class) as file:
            input_file = csv.DictReader(file)
            for row in input_file:
                filename_var = row['filename'].lower()
                class_var = row['class'].lower()
                label = class_to_int.get(class_var)
                if label is None:
                    class_to_int[class_var] = self.current_classes
                    self.current_classes += 1
                    label = class_to_int.get(class_var)
                self.class_dict[filename_var] = label

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
        path = self.paths[index % self.size]  # make sure index is within then range
        img = Image.open(path).convert('RGB')

        _, filename = os.path.split(path)

        label = self.class_dict.get(filename.lower())
        image = self.transform(img)

        return {'image': image, 'path': path, 'label': label}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size