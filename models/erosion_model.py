import torch
from .base_model import BaseModel
from .pix2pix_model import Pix2PixModel
from . import networks
import numpy as np


class ErosionModel(Pix2PixModel):
    def modify_commandline_options(parser, is_train=True):
        Pix2PixModel.modify_commandline_options(parser, is_train=is_train)
        parser.set_defaults(dataset_mode='erosion', input_nc=3, output_nc=1, preprocess='N.A.', image_type='uint16', image_value_bound=26350, no_flip=True)
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        return parser

    def __init__(self, opt):
        Pix2PixModel.__init__(self, opt)

    def set_input(self, input):
        super().set_input(input)
        #shape = list(self.real_A.shape)
        #self.gray_A = self.real_A[:, 1].view(shape[0], 1, shape[2], shape[3])

    def forward(self):
        self.input = self.real_A
        self.fake_B = self.netG(self.input)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.input, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        # Real
        real_B = self.real_B
        if self.opt.add_noise:
            z = (np.random.randn(*self.fake_B.size()).astype(np.float32) - 0.5) * 0.1
            z = torch.autograd.Variable(torch.from_numpy(z), requires_grad=False).to(self.device)
            real_B = real_B + z
        real_AB = torch.cat((self.input, real_B), 1)
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
        fake_AB = torch.cat((self.input, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        if self.opt.relativistic != 0:
            # Real
            real_AB = torch.cat((self.input, self.real_B), 1)
            pred_real = self.netD(real_AB).detach()
            self.loss_G_GAN = self.criterionGAN(pred_fake - torch.mean(pred_real) - 1, True)
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

    def compute_visuals(self, dataset=None):
        if not self.opt.fixed_example or dataset is None:
            return
        single = dataset.__getitem__(self.opt.fixed_index)
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = single['A' if AtoB else 'B'].unsqueeze(0).to(self.device)
        self.real_B = single['B' if AtoB else 'A'].unsqueeze(0).to(self.device)
        self.image_paths = [single['A_paths' if AtoB else 'B_paths']]
        self.forward()
