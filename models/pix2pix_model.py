import torch
from .base_model import BaseModel
from . import networks
import numpy as np


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
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
        parser.set_defaults(norm='batch', norm_G='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--progressive', action='store_true', help='progressive growing of networks')
        parser.add_argument('--progressive_stages', type=int, default=4, help='Number of stages in progression.')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--relativistic', type=int, default=0, help='relativistic loss')
            parser.add_argument('--add_noise', action='store_true', help='adds noise in discriminator training')
            parser.add_argument('--first_change_epoch', type=int, default=25, help='')
            parser.add_argument('--alpha_increase_interval', type=int, default=50, help='')
            parser.add_argument('--alpha_stabilize_interval', type=int, default=25, help='')

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
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, progressive=opt.progressive, progressive_stages=opt.progressive_stages, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, progressive=opt.progressive, progressive_stages=opt.progressive_stages, upsample_method=opt.upsample_method)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        elif opt.progressive:
            self.netG.module.update_alpha(1, opt.progressive_stages - 1)

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
        
        if self.opt.normG == 'adain':
            noise_inputs = []
            batch_size = self.real_A.size()[0]
            for length in self.netG.module.noise_length:
                z = (np.random.randn((batch_size, length)).astype(np.float32) - 0.5) / 0.5
                z = torch.autograd.Variable(torch.from_numpy(z), requires_grad=False).to(self.device)
                noise_inputs.append(z)
        else:
            self.noise_inputs = None

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if 'unet' in self.opt.netG:
            self.fake_B = self.netG(self.real_A, self.noise_inputs)
        else:
            self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        # Real
        real_B = self.real_B
        if self.opt.add_noise:
            z = (np.random.randn(*self.fake_B.size()).astype(np.float32) - 0.5) * 0.1
            z = torch.autograd.Variable(torch.from_numpy(z), requires_grad=False).to(self.device)
            real_B = real_B + z
        real_AB = torch.cat((self.real_A, real_B), 1)
        pred_real = self.netD(real_AB)

        if self.opt.relativistic != 0:
            self.loss_D_fake = self.criterionGAN(pred_fake - torch.mean(pred_real) + 1, False)
            self.loss_D_real = self.criterionGAN(pred_real - torch.mean(pred_fake) - 1, True)
        else:
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        if self.opt.relativistic != 0:
            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB).detach()
            self.loss_G_GAN = self.criterionGAN(pred_fake - torch.mean(pred_real) - 1, True) + self.criterionGAN(pred_real - torch.mean(pred_fake) + 1, False)
        else:
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
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

    def update_epoch_params(self, epoch):
        super().update_epoch_params(epoch)
        if not self.opt.progressive:
            return
        interval = self.opt.alpha_increase_interval + self.opt.alpha_stabilize_interval
        stop_point = self.opt.first_change_epoch + interval * (self.opt.progressive_stages - 1) - self.opt.alpha_stabilize_interval
        block = min(self.opt.progressive_stages - 1, (epoch + interval - self.opt.first_change_epoch - 1) // interval)
        alpha = 1 if (block == 0 or epoch >= stop_point) else min(1, ((epoch + interval - self.opt.first_change_epoch) % interval) / self.opt.alpha_increase_interval)
        self.netD.module.update_alpha(alpha, block)
        self.netG.module.update_alpha(alpha, block)
