import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os

class UnetErosionParametersModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', norm_G='instance', netG='unet_resblock', dataset_mode='exr', input_nc=3, output_nc=1, preprocess='N.A.', image_type='exr', no_flip=True, ngf=32)
        parser.add_argument('--get128', action='store_true', help='')
        parser.add_argument('--alpha_increase', type=float, default=0.005)
        parser.add_argument('--erosion_lr', type=float, default=0.005)
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
        self.visual_names += ['post_unet', 'fake_B']
        self.model_names = ['G', 'E']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, depth=opt.depth)
        self.netE = networks.init_net(networks.ErosionLayer(opt.width, opt.iterations, set_rain=True, no_parameters=True), gpu_ids=self.gpu_ids)
        self.tanh = torch.nn.Tanh()
        self.alpha = 0

        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.erosion_lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_E)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.flowmap = self.real_B[:, self.opt.output_flow_channel, :, :].unsqueeze(1).clone()
        self.image_paths = input['A_paths']

    def forward(self):
        if self.opt.get128:
            self.real_A = self.get_128(self.real_A)
            self.real_B = self.get_128(self.real_B)
            self.flowmap = self.get_128(self.flowmap)

        self.post_unet, latent = self.netG(self.real_A , True)
        self.post_unet = self.post_unet + self.real_A[:, self.opt.input_height_channel].unsqueeze(1)
        clamped = self.post_unet.squeeze(1)
        self.fake_B = torch.cat((torch.zeros_like(self.post_unet), self.netE(clamped, clamped, set_rain = (self.flowmap + 1) / 2, latent=latent, alpha=self.alpha).float()), 1)
        self.post_unet = torch.cat((torch.zeros_like(self.post_unet), self.post_unet), 1)
        self.fake_B = self.alpha * self.fake_B.clone() + (1 - self.alpha) * self.post_unet

    def backward_D(self):
        self.loss_D = torch.zeros([1]).to(self.device)

    def backward_G(self):
        self.loss_G = self.criterionL2(self.fake_B, self.real_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.backward_D()
        self.optimizer_G.zero_grad()
        self.optimizer_E.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_E.step()

    def compute_visuals(self, dataset=None):
        if not self.opt.fixed_example or dataset is None:
            return
        single = dataset.dataset.get_val_item(self.opt.fixed_index)
        self.real_A = single['A'].unsqueeze(0).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        self.real_B = single['B'].unsqueeze(0).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        self.flowmap = self.real_B[:, self.opt.output_flow_channel, :, :].clone().repeat(len(self.gpu_ids), 1, 1, 1)
        self.image_paths = [single['A_paths']]

        self.forward()
        loss_G = self.criterionL2(self.fake_B, self.fake_B)
        loss_G.backward()

    def update_epoch_params(self, epoch):
        super().update_epoch_params(epoch)
        self.alpha = min(1, epoch * self.opt.alpha_increase)

    def get_128(self, image):
        batch_size = image.shape[0]
        return self.break_into_16(image)[:batch_size]

    def break_into_16(self, image):
        return self.break_into_4(self.break_into_4(image))

    def break_into_4(self, image):
        return torch.cat(torch.chunk(torch.cat(torch.chunk(image, 2, dim=2), 0), 2, dim=3), 0)

    def combine_from_16(self, image):
        return self.combine_from_4(self.combine_from_4(image))

    def combine_from_4(self, image):
        return torch.cat(torch.chunk(torch.cat(torch.chunk(image, 2, dim=0), 3), 2, dim=0), 2)