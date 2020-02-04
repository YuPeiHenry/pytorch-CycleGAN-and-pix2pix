import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os

class HRnetFlowmapModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(netG='hrnet', dataset_mode='exr', image_type='exr', input_nc=3, output_nc=1)
        parser.add_argument('--exclude_input', action='store_true', help='')
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G']
        self.visual_names = ['real_A'] if not opt.exclude_input else []
        self.visual_names += ['fake_B']
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method)

        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)[:, 0].unsqueeze(1)
        self.image_paths = input['A_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        self.loss_D = torch.zeros([1]).to(self.device)

    def backward_G(self):
        self.loss_G = self.criterionL2(self.fake_B, self.real_B)
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
        self.real_B = single['B'].unsqueeze(0).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        self.image_paths = [single['A_paths']]

        self.forward()
        self.fake_B = torch.cat((self.fake_B, torch.zeros_like(self.fake_B)), 1)
