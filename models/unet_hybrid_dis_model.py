import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os

class UnetHybridDisModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', norm_G='instance', netG='unet_resblock', dataset_mode='exr', input_nc=3, output_nc=1, preprocess='N.A.', image_type='exr', no_flip=True, ngf=32)
        parser.add_argument('--negative_constant', type=float, default=4.0, help='')
        parser.add_argument('--L1', action='store_true', help='')
        parser.add_argument('--exclude_input', action='store_true', help='')
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        parser.add_argument('--depth', type=int, default=6, help='')
        parser.add_argument('--input_height_channel', type=int, default=0)
        parser.add_argument('--output_height_channel', type=int, default=1)
        parser.add_argument('--output_flow_channel', type=int, default=0)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G', 'D']
        self.visual_names = ['real_A', 'real_B'] if not opt.exclude_input else []
        self.visual_names += ['fake_B']
        self.model_names = ['G', 'D']
        self.sigmoid = torch.nn.Sigmoid();
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, depth=opt.depth)
        self.netD = networks.define_G(1, 1, opt.ngf, opt.netG, opt.norm_G,
                                  not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, depth=opt.depth - 2)

        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.A_orig = input['A_orig'][:, self.opt.input_height_channel, :, :].unsqueeze(1).to(self.device)
        # Account for saving the full heightmap instead of residual
        self.B_orig = input['B_orig'][:, self.opt.output_height_channel, :, :].to(self.device).unsqueeze(1)
        self.flowmap = self.real_B[:, self.opt.output_flow_channel, :, :].unsqueeze(1).clone()
        self.image_paths = input['A_paths']

    def forward(self):
        """
        if self.opt.break4:
            self.real_A = self.break_into_4(self.real_A)
            self.real_B = self.break_into_4(self.real_B)
        """
        self.fake_B = self.netG(self.real_A)
        self.fake_B = self.fake_B + self.A_orig
        self.flow_mult = self.sigmoid(self.netD(self.flowmap))
        self.flow_mult = (self.flow_mult - torch.mean(self.flow_mult, dim=0) * 0.9)

        if not self.isTrain:
            self.fake_B = self.fake_B * 2

    def backward_D(self):
        self.loss_D = -self.criterionL2(self.flow_mult * self.fake_B.detach(), self.flow_mult * self.B_orig)
        self.loss_D.backward()

    def backward_G(self):
        self.loss_G = self.criterionL2(self.flow_mult.detach() * self.fake_B, self.flow_mult.detach() * self.B_orig)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def compute_visuals(self, dataset=None):
        if not self.opt.fixed_example or dataset is None:
            return
        single = dataset.dataset.get_val_item(self.opt.fixed_index)
        self.real_A = single['A'].unsqueeze(0).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        self.real_B = single['B'].unsqueeze(0).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        self.A_orig = single['A_orig'].unsqueeze(0)[:, self.opt.input_height_channel, :, :].unsqueeze(1).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        self.B_orig = single['B_orig'].unsqueeze(0)[:, self.opt.output_height_channel, :, :].unsqueeze(1).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        self.image_paths = [single['A_paths']]

        self.forward()
        loss_G = self.criterionL2(self.fake_B, self.B_orig)
        loss_G.backward()
        loss_D = -self.criterionL2(self.flow_mult * self.fake_B.detach(), self.flow_mult * self.B_orig.detach())
        loss_D.backward()
        self.fake_B = self.fake_B * 2
        """
        if self.opt.break4:
            self.real_A = self.combine_from_4(self.real_A)
            self.real_B = self.combine_from_4(self.real_B)
            self.post_unet = self.combine_from_4(self.post_unet)
        """

    def break_into_4(self, image):
        return torch.cat(torch.chunk(torch.cat(torch.chunk(image, 2, dim=2), 0), 2, dim=3), 0)

    def combine_from_4(self, image):
        return torch.cat(torch.chunk(torch.cat(torch.chunk(image, 2, dim=0), 3), 2, dim=0), 2)