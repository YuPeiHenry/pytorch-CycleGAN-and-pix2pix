import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class InstanceDataset(BaseDataset):
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
        self.A = os.path.join(opt.dataroot, opt.phase + '_A')  # create a path '/path/to/data/train_A'
        self.inst = os.path.join(opt.dataroot, opt.phase + '_inst')  # create a path '/path/to/train_inst'
        self.B = os.path.join(opt.dataroot, opt.phase + '_B')  # create a path '/path/to/train_B'

        self.A_paths = sorted(make_dataset(self.A, opt.max_dataset_size))
        self.inst_paths = sorted(make_dataset(self.inst, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.inst_size = len(self.inst_paths)
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_inst = get_transform(self.opt, grayscale=True)
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

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
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        inst_path = self.B_paths[index % self.inst_size]
        B_path = self.B_paths[index % self.B_size]
        A_img = Image.open(A_path).convert('RGB')
        inst_img = Image.open(inst_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        inst = self.transform_inst(inst_img)
        B = self.transform_B(B_img)

        return {'A': A, 'inst': inst, 'B': B, 'A_paths': A_path, 'inst_paths': inst_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
