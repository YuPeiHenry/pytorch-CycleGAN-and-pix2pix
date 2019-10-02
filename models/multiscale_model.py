import torch
from .base_model import BaseModel
from . import networks
import numpy as np


class MultiscaleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='erosion', input_nc=3, output_nc=1, preprocess='N.A.', image_type='uint16', image_value_bound=26350, no_flip=True)
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--relativistic', type=int, default=0, help='relativistic loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'multi_unet', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multi_n_layers',
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, upsample_method=opt.upsample_method)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_Bs = self.netG(self.real_A)  # G(A)
        self.fake_B = self.fake_Bs[0]

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        real_As = self.netD(self.real_A, create_series=True)
        
        fake_ABs = self.create_AB(real_As, self.fake_Bs, detach=True)
        pred_fake = self.netD(fake_ABs)
        real_Bs = self.netD(self.real_B, create_series=True)
        real_ABs = self.create_AB(real_As, real_Bs)
        pred_real = self.netD(real_ABs)

        self.loss_D_fake = 0
        self.loss_D_real = 0
        if self.opt.relativistic != 0:
            for pred_real_elem, pred_fake_elem in zip(pred_real, pred_fake):
                self.loss_D_fake += self.criterionGAN(pred_fake_elem - torch.mean(pred_real_elem) + 1, False)
                self.loss_D_real += self.criterionGAN(pred_real_elem - torch.mean(pred_fake_elem) - 1, True)
        else:
            for pred_real_elem, pred_fake_elem in zip(pred_real, pred_fake):
                self.loss_D_fake += self.criterionGAN(pred_fake_elem, False)        
                self.loss_D_real += self.criterionGAN(pred_real_elem, True)
        
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        real_As = self.netD(self.real_A, create_series=True)
        
        fake_ABs = self.create_AB(real_As, self.fake_Bs)
        pred_fake = self.netD(fake_ABs)
        real_Bs = self.netD(self.real_B, create_series=True)
        real_ABs = self.create_AB(real_As, real_Bs, detach=True)
        pred_real = self.netD(real_ABs)

        self.loss_G_GAN = 0
        if self.opt.relativistic != 0:
            for pred_real_elem, pred_fake_elem in zip(pred_real, pred_fake):
                self.loss_G_GAN += self.criterionGAN(pred_fake_elem - torch.mean(pred_real_elem) - 1, True)
        else:
            for pred_real_elem, pred_fake_elem in zip(pred_real, pred_fake):
                self.loss_G_GAN += self.criterionGAN(pred_fake_elem, True)

        self.loss_G_L1 = 0
        for real_B, fake_B in zip(real_Bs, self.fake_Bs):
            self.loss_G_L1 += self.criterionL1(fake_B, real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def update_learning_rate(self):
        super().update_learning_rate()

    def compute_visuals(self, dataset=None):
        if not self.opt.fixed_example or dataset is None:
            return
        single = dataset.__getitem__(self.opt.fixed_index)
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = single['A' if AtoB else 'B'].unsqueeze(0).to(self.device)
        self.real_B = single['B' if AtoB else 'A'].unsqueeze(0).to(self.device)
        self.image_paths = [single['A_paths' if AtoB else 'B_paths']]
        self.forward()

    def create_AB(As, Bs, detach=False):
        AB = []
        for x, y in zip(A, B):
            if detach:
                AB.append(torch.cat((x, y), 1).detach())
            else:
                AB.append(torch.cat((x, y), 1))
        return AB
