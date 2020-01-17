import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os

class EncoderModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', norm_G='instance', netG='unet_resblock', dataset_mode='exr_one_channel', input_nc=1, output_nc=1, preprocess='N.A.', image_type='exr', no_flip=True, ngf=32)
        parser.add_argument('--exclude_input', action='store_true', help='')
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        parser.add_argument('--depth', type=int, default=5, help='')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G']
        self.visual_names = ['real_A'] if not opt.exclude_input else []
        self.visual_names += ['fake_A']
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, depth=opt.depth)

        if self.isTrain:
            self.downsample = nn.AvgPool2d(kernel_size=4, stride=4, padding=0, ceil_mode=False)
            self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.A_orig = input['A_orig'].to(self.device)
        self.A_blur = self.upsample(self.downsample(self.A_orig))
        self.image_paths = input['A_paths']

    def forward(self):
        """
        if self.opt.break4:
            self.real_A = self.break_into_4(self.real_A)
            self.real_B = self.break_into_4(self.real_B)
        """
        self.fake_A = self.netG(self.real_A)

    def backward_D(self):
        self.loss_D = torch.zeros([1]).to(self.device)

    def backward_G(self):
        self.loss_G = self.criterionL2(self.fake_A + self.A_blur, self.real_A)
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
        self.real_A = single['A'].unsqueeze(0).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        self.image_paths = [single['A_paths']]

        self.forward()

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
