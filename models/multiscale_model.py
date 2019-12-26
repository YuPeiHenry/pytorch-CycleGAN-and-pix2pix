import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
import numpy as np


class MultiscaleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', norm_G='instance', netG='unet_256', dataset_mode='exr_height', input_nc=3, output_nc=1, preprocess='N.A.', image_type='exr', image_value_bound=26350, no_flip=True)
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'multi_unet', opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method)


        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        self.decimation = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        o1_min = torch.Tensor([-1., -1.]).to(self.device).view(1, 2, 1, 1)
        o1_max = torch.Tensor([1., 1.]).to(self.device).view(1, 2, 1, 1)
        o2_min = torch.Tensor([-1.1, -0.06]).to(self.device).view(1, 2, 1, 1)
        o2_max = torch.Tensor([1., 0.05]).to(self.device).view(1, 2, 1, 1)
        o3_min = torch.Tensor([-1.1, -0.07]).to(self.device).view(1, 2, 1, 1)
        o3_max = torch.Tensor([1.1, 0.06]).to(self.device).view(1, 2, 1, 1)
        
        self.o_channels_min = [o1_min, o2_min, o3_min]
        self.o_channels_max = [o1_max, o2_max, o3_max]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        real_Bs = [self.real_B]
        for _ in range(2):
            real_Bs.append(self.decimation(real_Bs[-1]))
        real_Bs.reverse()
        self.real_Bs = [real_Bs[0]]
        for i in range(1, 3):
            value = real_Bs[i] - self.upsample(real_Bs[i - 1])
            value = value - (self.o_channels_max[i] + self.o_channels_min[i]) / 2
            value = value / (self.o_channels_max[i] - self.o_channels_min[i]) * 2
            self.real_Bs.append(value)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_Bs = self.netG(self.real_A)  # G(A)
        self.fake_B = self.create_fakeB(self.fake_Bs)

    def backward_D(self):
        self.loss_D = torch.zeros([1]).to(self.device)

    def backward_G(self):
        self.loss_G = torch.zeros([1]).to(self.device)
        self.loss_G_L2 = 0
        for real_B, fake_B in zip(self.real_Bs, self.fake_Bs):
            self.loss_G_L2 = self.loss_G_L2 + self.criterionL2(fake_B, real_B) * 1000
        self.loss_G = self.loss_G_L2 / 1000
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.backward_D()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def compute_visuals(self, dataset=None):
        if not self.opt.fixed_example or dataset is None:
            return
        single = dataset.dataset.get_val_item(self.opt.fixed_index)
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = single['A' if AtoB else 'B'].unsqueeze(0).to(self.device)
        self.real_B = single['B' if AtoB else 'A'].unsqueeze(0).to(self.device)
        self.image_paths = [single['A_paths' if AtoB else 'B_paths']]

        self.forward()

    def create_fakeB(self, fake_Bs):
        lvls = len(fake_Bs)
        fakeB = fake_Bs[0]
        for i in range(1, lvls):
            value = fake_Bs[i]
            value = value * (self.o_channels_max[i] - self.o_channels_min[i]) / 2
            value = value + (self.o_channels_max[i] + self.o_channels_min[i]) / 2
            fakeB = self.upsample(fakeB) + value
        return fakeB
