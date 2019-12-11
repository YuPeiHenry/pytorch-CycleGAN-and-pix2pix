import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import numpy as np
import torch
import data.exrlib as exrlib

class ExrDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        assert(opt.image_type == 'exr')

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

        self.input_channels = [3, 4, 6, 7] #height, mesa, slopex, slopez
        self.output_channels = [5, 6, 7, 9, 10] #flowx, flowz, height, sediment, water

        if not self.opt.compute_bounds:
            self.i_channels_min = np.array([[[0, 0, -274.83758545, -317.95968628]]])
            self.i_channels_max = np.array([[[417.75454712, 0.99980736, 10.71749306, 10.45528221]]])
            self.o_channels_min = np.array([[[-10.80896282, -12.17011833, -3.95739818, 0, -0.87560517]]])
            self.o_channels_max = np.array([[[2.76426649, 2.35645199, 418.50262451, 4.86357594, 2.89976954]]])
            return

        channels_min = np.array([2**16 for _ in self.input_channels])
        channels_max = np.array([0 for _ in self.input_channels])
        examples = 0
        for A1_path in self.A1_paths:
            A1_img = exrlib.read_exr(A1_path)[0][:, :, self.input_channels].transpose(2, 0, 1).reshape(len(self.input_channels), -1)
            channels_min = np.min(np.concatenate((np.expand_dims(channels_min, 1), np.expand_dims(np.min(A1_img, 1), 1)), 1), 1)
            channels_max = np.max(np.concatenate((np.expand_dims(channels_min, 1), np.expand_dims(np.max(A1_img, 1), 1)), 1), 1)
            examples += 1
            if examples >= 1000:
                break

        print(channels_min)
        self.i_channels_min = np.expand_dims(np.expand_dims(np.array(channels_min), 1), 2)
        print(channels_max)
        self.i_channels_max = np.expand_dims(np.expand_dims(np.array(channels_max), 1), 2)

        channels_min = np.array([2**16 for _ in self.output_channels])
        channels_max = np.array([0 for _ in self.output_channels])
        examples = 0
        for B_path in self.B_paths:
            B_img = exrlib.read_exr(B_path)[0][:, :, self.output_channels].transpose(2, 0, 1).reshape(len(self.output_channels), -1)
            channels_min = np.min(np.concatenate((np.expand_dims(channels_min, 1), np.expand_dims(np.min(B_img, 1), 1)), 1), 1)
            channels_max = np.max(np.concatenate((np.expand_dims(channels_min, 1), np.expand_dims(np.max(B_img, 1), 1)), 1), 1)
            examples += 1
            if examples >= 1000:
                break

        print(channels_min)
        self.o_channels_min = np.expand_dims(np.expand_dims(channels_min, 1), 2)
        print(channels_max)
        self.o_channels_max = np.expand_dims(np.expand_dims(channels_max, 1), 2)

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
        A1_img = exrlib.read_exr(A1_path)[0][:, :, self.input_channels]
        B_img = exrlib.read_exr(B_path)[0][:, :, self.output_channels]
        #A = torch.Tensor(A_img)
        A1 = self.convert_input(A1_img)
        B = self.convert_output(B_img)

        return {'A': A1, 'B': B, 'A_paths': A1_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A1_size, self.B_size)

    def convert_input(self, image):
        image = image - (self.i_channels_max + self.i_channels_min) / 2
        image = image / (self.i_channels_max - self.i_channels_min) * 2
        image = np.transpose(image, (2, 0, 1))
        return torch.Tensor(image)

    def convert_output(self, image):
        image = image - (self.o_channels_max + self.o_channels_min) / 2
        image = image / (self.o_channels_max - self.o_channels_min) * 2
        image = np.transpose(image, (2, 0, 1))
        return torch.Tensor(image)

    def convert_output_to_image(self, output_arr):
        if output_arr.shape[2] == self.output_channels:
            output_arr = output_arr * (self.o_channels_max - self.o_channels_min) / 2
            output_arr = output_arr + (self.o_channels_max + self.o_channels_min) / 2
            return output_arr.astype(np.float32)
        else:
            output_arr = output_arr * (self.i_channels_max - self.i_channels_min) / 2
            output_arr = output_arr + (self.i_channels_max + self.i_channels_min) / 2
            return output_arr.astype(np.float32)

    def write(self):
        write_exr(image_path, image, [str(i) for i in range(image.shape[2])])