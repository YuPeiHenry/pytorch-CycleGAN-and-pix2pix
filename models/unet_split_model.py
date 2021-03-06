import torch
from .base_model import BaseModel
from .erosionlib import *
from . import networks
import numpy as np
import os

class UnetSplitModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', norm_G='instance', netG='unet_resblock_split', dataset_mode='exr', input_nc=1, output_nc=1, preprocess='N.A.', image_type='exr', no_flip=True, ngf=32)
        parser.add_argument('--get256', action='store_true', help='')
        parser.add_argument('--full_psnr', action='store_true', help='')
        parser.add_argument('--exclude_input', action='store_true', help='')
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        parser.add_argument('--width', type=int, default=512)
        parser.add_argument('--iterations', type=int, default=10)
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
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, depth=opt.depth)

        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.var_names = ['val']
            self.var_values = [0]
            self.var_grads = [0]

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.input_height = self.real_A[:, self.opt.input_height_channel].unsqueeze(1)
        #self.flowmap = self.real_B[:, self.opt.output_flow_channel, :, :].unsqueeze(1).clone()
        self.image_paths = input['A_paths']

    def forward(self):
        if self.opt.get256:
            self.real_A = self.get_256(self.real_A)
            self.real_B = self.get_256(self.real_B)
            self.input_height = self.get_256(self.input_height)
            #self.flowmap = self.get_256(self.flowmap)

        slope = simple_gradient(self.input_height.squeeze(1), torch.zeros_like(self.input_height).squeeze(1), 1e-10)
        slope_magnitude = (slope[:, :, :, 0] ** 2 + slope[:, :, :, 1] ** 2).unsqueeze(1)
        self.fake_B = self.netG(self.input_height, slope_magnitude)
        self.fake_B = self.fake_B / 10 + self.input_height
        self.fake_B = torch.cat((torch.zeros_like(self.fake_B), self.fake_B), 1)
        self.target = self.real_B[:, self.opt.output_height_channel].unsqueeze(1)
        self.target = torch.cat((torch.zeros_like(self.target), self.target.clone()), 1)

    def backward_D(self):
        self.loss_D = torch.zeros([1]).to(self.device)

    def backward_G(self):
        if self.opt.full_psnr:
            diff = self.real_B[:, self.opt.output_height_channel].unsqueeze(1) - self.input_height
            self.loss_G = -20 * torch.log10(diff / torch.sqrt(torch.mean((self.fake_B - self.target) ** 2)))
        else:
            self.loss_G = torch.log(self.criterionL2(self.fake_B, self.target))
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
        self.input_height = self.real_A[:, self.opt.input_height_channel].unsqueeze(1)
        #self.flowmap = self.real_B[:, self.opt.output_flow_channel, :, :].clone().repeat(len(self.gpu_ids), 1, 1, 1)
        self.image_paths = [single['A_paths']]

        self.forward()
        loss_G = self.criterionL2(self.fake_B, self.target)
        self.var_values = [loss_G.item()]
        loss_G.backward()

    def get_256(self, image):
        batch_size = image.shape[0]
        return self.break_into_4(image)[0].unsqueeze(0)

    def break_into_16(self, image):
        return self.break_into_4(self.break_into_4(image))

    def break_into_4(self, image):
        return torch.cat(torch.chunk(torch.cat(torch.chunk(image, 2, dim=2), 0), 2, dim=3), 0)

    def combine_from_16(self, image):
        return self.combine_from_4(self.combine_from_4(image))

    def combine_from_4(self, image):
        return torch.cat(torch.chunk(torch.cat(torch.chunk(image, 2, dim=0), 3), 2, dim=0), 2)