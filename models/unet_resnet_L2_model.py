import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os


class UnetResnetL2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', norm_G='instance', netG='resnet_9blocks', dataset_mode='exr', input_nc=5, output_nc=2, preprocess='N.A.', image_type='exr', image_value_bound=26350, no_flip=True)
        parser.add_argument('--unet_residue', action='store_true', help='')
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        parser.add_argument('--netU', type=str, default='unet_256')
        parser.add_argument('--unet_input_nc', type=int, default=3)
        parser.add_argument('--use_feature_extractor', action='store_true', help='')
        parser.add_argument('--break4', action='store_true', help='')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_L2', 'G_L2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'post_unet', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        if opt.use_feature_extractor: self.model_names += ['Feature']
        self.preload_names = ['U']
        # define networks
        self.netU = networks.define_G(opt.unet_input_nc, opt.output_nc, opt.ngf, opt.netU, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, linear=opt.linear)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, linear=opt.linear)
        if opt.use_feature_extractor:
            self.netFeature = networks.init_net(networks.FeatureExtractor(opt.output_nc), gpu_ids=self.gpu_ids)
        self.load_base_networks()

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if opt.use_feature_extractor:
                self.optimizer_Feature = torch.optim.Adam(self.netFeature.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_Feature)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_B = 1 - torch.nn.ReLU()(2 - torch.nn.ReLU()(self.real_B + 1)) #clip to [-1, 1]
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.break4:
            self.real_A = self.break_into_4(self.real_A)
            self.real_B = self.break_into_4(self.real_B)
        self.post_unet = self.netU(self.real_A).detach()
        if self.opt.unet_residue:
            self.post_unet[:, 1, :, :] = self.post_unet[:, 1, :, :] + self.real_A[:, 0, :, :]
        self.fake_B = self.netG(torch.cat((self.real_A, self.post_unet), 1))
        self.fake_B = self.fake_B + self.post_unet

    def backward_D(self):
        if not self.opt.use_feature_extractor:
            self.loss_D_L2 = torch.zeros([1]).to(self.device)
            self.loss_D = self.loss_D_L2
        else:
            fake_B_features = self.netFeature(self.fake_B.detach())
            real_features = self.netFeature(self.real_B)
            self.loss_D_L2 = -self.criterionL2(fake_B_features, real_features) * 1000
            self.loss_D = -self.loss_D_L2 / self.loss_D_L2.item() * 2 / 1000
            self.loss_D.backward()

    def backward_G(self):
        self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * 1000
        if not self.opt.use_feature_extractor:
            self.loss_G = self.loss_G_L2 / 1000
        else:
            fake_B_output = self.netFeature(self.fake_B)
            real_B_output = self.netFeature(self.real_B)
            feat_loss = self.criterionL2(fake_B_output, real_B_output)
            self.loss_G = self.loss_G_L2 / 1000 + feat_loss / feat_loss.item() * self.loss_G_L2.item() / 1000
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        if self.opt.use_feature_extractor:
            self.set_requires_grad(self.netFeature, True)
            self.optimizer_Feature.zero_grad()
        self.backward_D()
        if self.opt.use_feature_extractor:
            self.optimizer_Feature.step()
            self.set_requires_grad(self.netFeature, False)
        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def compute_visuals(self, dataset=None):
        if not self.opt.fixed_example or dataset is None:
            return
        single = dataset.dataset.get_val_item(self.opt.fixed_index)
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = single['A' if AtoB else 'B'].unsqueeze(0).to(self.device)
        self.real_B = single['B' if AtoB else 'A'].unsqueeze(0).to(self.device)
        self.image_paths = [single['A_paths' if AtoB else 'B_paths']]

        self.forward()
        if self.opt.break4:
            self.real_A = self.combine_from_4(self.real_A)
            self.real_B = self.combine_from_4(self.real_B)
            self.post_unet = self.combine_from_4(self.post_unet)
            self.fake_B = self.combine_from_4(self.fake_B)

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
