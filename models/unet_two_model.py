import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os

class UnetTwoModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', norm_G='instance', netG='unet_resblock', dataset_mode='exr', input_nc=3, output_nc=2, preprocess='N.A.', image_type='exr', no_flip=True, ngf=32)
        parser.add_argument('--exclude_input', action='store_true', help='')
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        parser.add_argument('--depth', type=int, default=6, help='')
        parser.add_argument('--input_height_channel', type=int, default=0)
        parser.add_argument('--output_height_channel', type=int, default=1)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G']
        self.visual_names = ['real_A'] if not opt.exclude_input else []
        self.visual_names += ['fake_B']
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc + opt.output_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, depth=opt.depth)
        self.preload_names = ['E']
        self.netE = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, depth=opt.depth)
        self.load_base_networks()
        self.netE.eval()

        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.A_orig = input['A_orig'][:, self.opt.input_height_channel, :, :].unsqueeze(1).to(self.device)
        self.B_orig = self.real_B.clone()
        self.B_orig[:, self.opt.output_height_channel, :, :] = input['B_orig'][:, self.opt.output_height_channel, :, :].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """
        if self.opt.break4:
            self.real_A = self.break_into_4(self.real_A)
            self.real_B = self.break_into_4(self.real_B)
        """
        in_h = self.opt.input_height_channel
        out_h = self.opt.output_height_channel
        residue_A = self.netE(self.real_A)
        self.fake_B = self.netG(torch.cat((self.real_A, residue_A / 10), 1))
        residue = torch.zeros(self.fake_B.shape).to(self.device)
        residue[:, out_h, :, :] = self.A_orig.squeeze(1) + residue_A[:, out_h, :, :]
        self.fake_B = self.fake_B + residue

    def backward_D(self):
        self.loss_D = torch.zeros([1]).to(self.device)

    def backward_G(self):
        self.loss_G = self.criterionL2(self.fake_B, self.B_orig)
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
        self.A_orig = single['A_orig'].unsqueeze(0)[:, self.opt.input_height_channel, :, :].unsqueeze(1).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        
        self.image_paths = [single['A_paths']]

        self.forward()
        self.fake_B[:, self.opt.output_height_channel, :, :] = self.fake_B[:, self.opt.output_height_channel, :, :] - ((910 - 86) / 2)
        self.fake_B[:, self.opt.output_height_channel, :, :] = self.fake_B[:, self.opt.output_height_channel, :, :] / (910 + 86) * 2
        """
        if self.opt.break4:
            self.real_A = self.combine_from_4(self.real_A)
            self.real_B = self.combine_from_4(self.real_B)
            self.post_unet = self.combine_from_4(self.post_unet)
        """
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

    def break_into_4(self, image):
        return torch.cat(torch.chunk(torch.cat(torch.chunk(image, 2, dim=2), 0), 2, dim=3), 0)

    def combine_from_4(self, image):
        return torch.cat(torch.chunk(torch.cat(torch.chunk(image, 2, dim=0), 3), 2, dim=0), 2)
