import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os

class HRnetErosionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(netG='unet_resblock', dataset_mode='exr', image_type='exr', input_nc=3, output_nc=1)
        parser.add_argument('--netF', type=str, default='hrnet')
        parser.add_argument('--width', type=int, default=512)
        parser.add_argument('--iterations', type=int, default=10)
        parser.add_argument('--exclude_input', action='store_true', help='')
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        parser.add_argument('--erosion_lr', type=float, default=0.2)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G']
        self.visual_names = ['real_A', 'real_B'] if not opt.exclude_input else []
        self.visual_names += ['fake_B']
        self.model_names = ['F', 'G', 'Erosion']
        self.preload_names = ['F']
        self.netF = networks.define_G(opt.input_nc, 1, opt.ngf, opt.netF, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method)
        self.netG = networks.define_G(opt.input_nc, 1, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method)
        self.netErosion = networks.init_net(networks.ErosionLayer(opt.width, opt.iterations, set_rain=True), gpu_ids=self.gpu_ids)

        self.load_base_networks()
        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_F)
            self.optimizer_Erosion = torch.optim.Adam(self.netErosion.parameters(), lr=opt.erosion_lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Erosion)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)[:, 1].unsqueeze(1)
        self.image_paths = input['A_paths']

    def forward(self):
        self.flowmap = self.netF(self.real_A)
        heightmap = self.netG(self.real_A) + self.real_A[:, 0].unsqueeze(1)
        self.fake_B = self.netErosion(heightmap, heightmap, set_rain=(self.flowmap + 1) / 2).float()
        
        if not self.isTrain:
            self.fake_B = torch.cat((self.flowmap, self.fake_B), 1)

    def backward_D(self):
        self.loss_D = torch.zeros([1]).to(self.device)

    def backward_G(self):
        self.loss_G = self.criterionL2(self.fake_B, self.real_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.backward_D()
        self.optimizer_F.zero_grad()
        self.optimizer_G.zero_grad()
        self.optimizer_Erosion.zero_grad()
        self.backward_G()
        self.optimizer_F.step()
        self.optimizer_G.step()
        self.optimizer_Erosion.step()

    def compute_visuals(self, dataset=None):
        if not self.opt.fixed_example or dataset is None:
            return
        single = dataset.dataset.get_val_item(self.opt.fixed_index)
        self.real_A = single['A'].unsqueeze(0).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        self.real_B = single['B'].unsqueeze(0).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        self.image_paths = [single['A_paths']]

        self.forward()
        loss_G = self.criterionL2(self.fake_B, self.real_B)
        loss_G.backward()
        self.fake_B = torch.cat((self.flowmap, self.fake_B), 1)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_base_networks(self):
        for name in self.preload_names:
            if isinstance(name, str):
                load_filename = 'base_net_%s.pth' % (name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
