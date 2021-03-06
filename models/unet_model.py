import torch
from .base_model import BaseModel
from . import networks
from collections import OrderedDict
import numpy as np
import os
import data.exrlib as exrlib

class UnetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', norm_G='instance', netG='unet_256', dataset_mode='exr', input_nc=3, output_nc=2, preprocess='N.A.', image_type='exr', image_value_bound=26350, no_flip=True)
        parser.add_argument('--generate_residue', action='store_true', help='')
        parser.add_argument('--downsampleConv', type=int, default=0)
        parser.add_argument('--upsampleConv', type=int, default=0)
        parser.add_argument('--maxFilters', type=int, default=512)

        parser.add_argument('--SGD', action='store_true', help='')
        parser.add_argument('--log_error', action='store_true', help='')
        parser.add_argument('--input_height_channel', type=int, default=0)
        parser.add_argument('--output_height_channel', type=int, default=1)
        parser.add_argument('--output_flow_channel', type=int, default=0)
        parser.add_argument('--exclude_input', action='store_true', help='')
        parser.add_argument('--exclude_flowmap', action='store_true', help='')
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        parser.add_argument('--half_lr', action='store_true', help='')
        parser.add_argument('--width', type=int, default=512)
        parser.add_argument('--iterations', type=int, default=10)
        parser.add_argument('--preload_unet', action='store_true', help='')
        parser.add_argument('--temp_unet_fix', action='store_true', help='')
        parser.add_argument('--use_erosion', action='store_true', help='')
        parser.add_argument('--use_rain_conv', action='store_true', help='')
        parser.add_argument('--blend_inputs', action='store_true', help='')
        parser.add_argument('--erosion_random', action='store_true', help='')
        parser.add_argument('--erosion_flowmap', action='store_true', help='')
        parser.add_argument('--erosion_only', action='store_true', help='')
        parser.add_argument('--store_water', action='store_true', help='')
        parser.add_argument('--debug_gradients', action='store_true', help='')
        parser.add_argument('--lambda_L1', type=float, default=0.0, help='')
        parser.add_argument('--lambda_L2', type=float, default=1.0, help='')
        parser.add_argument('--erosion_lr', type=float, default=0.0001, help='')
        parser.add_argument('--use_feature_extractor', action='store_true', help='')
        parser.add_argument('--break4', action='store_true', help='')
        parser.add_argument('--disable_cudnn', action='store_true', help='')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        if opt.disable_cudnn: torch.backends.cudnn.enabled = False
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_L2', 'G_L2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B'] if not opt.exclude_input else []
        if opt.use_erosion: self.visual_names += ['fake_B']
        if not opt.erosion_only: self.visual_names += ['post_unet']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        if opt.use_erosion: self.model_names += ['Erosion']
        if opt.use_feature_extractor: self.model_names += ['Feature']
        self.preload_names = []
        # define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, linear=opt.linear, numDownsampleConv=opt.downsampleConv, numUpsampleConv=opt.upsampleConv, max_filters=opt.maxFilters)
        if opt.use_erosion:
            self.netErosion = networks.init_net(networks.ErosionLayer(opt.width, opt.iterations, opt.erosion_flowmap, opt.erosion_random, opt.blend_inputs, opt.use_rain_conv), gpu_ids=self.gpu_ids)
        if opt.preload_unet:
            self.preload_names += ['G']
            self.set_requires_grad(self.netG, False)
        if opt.use_feature_extractor:
            self.netFeature = networks.init_net(networks.FeatureExtractor(opt.output_nc), gpu_ids=self.gpu_ids)
        self.load_base_networks()

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        if self.isTrain and opt.SGD:
            self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9)
            if not opt.preload_unet: self.optimizers.append(self.optimizer_G)
            if opt.use_erosion:
                self.optimizer_Erosion = torch.optim.SGD(self.netErosion.parameters(), lr=opt.erosion_lr, momentum=0.9)
                self.optimizers.append(self.optimizer_Erosion)
            if opt.use_feature_extractor:
                self.optimizer_Feature = torch.optim.SGD(self.netFeature.parameters(), lr=opt.lr, momentum=0.9)
                self.optimizers.append(self.optimizer_Feature)
        elif self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if not opt.preload_unet:
                self.optimizers.append(self.optimizer_G)
            else:
                self.netG.eval()
            if opt.use_erosion:
                self.optimizer_Erosion = torch.optim.Adam(self.netErosion.parameters(), lr=opt.erosion_lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_Erosion)
            if opt.use_feature_extractor:
                self.optimizer_Feature = torch.optim.Adam(self.netFeature.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_Feature)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        if self.opt.exclude_flowmap: self.real_B = self.real_B[:, 1, :, :].unsqueeze(1)
        out_h = self.opt.output_height_channel if not self.opt.exclude_flowmap else 1
        if self.opt.linear:
            self.residue = input['A_orig'][:, self.opt.input_height_channel, :, :].to(self.device)
            if not self.opt.use_erosion: self.real_B[:, self.opt.output_height_channel, :, :] = input['B_orig'][:, out_h, :, :].to(self.device)
        else:
            self.real_B = 1 - torch.nn.ReLU()(2 - torch.nn.ReLU()(self.real_B + 1)) #clip to [-1, 1]
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.break4:
            self.real_A = self.break_into_4(self.real_A)
            self.real_B = self.break_into_4(self.real_B)
        in_h = self.opt.input_height_channel
        out_h = self.opt.output_height_channel
        out_f = self.opt.output_flow_channel
        if not self.opt.erosion_only:
            self.post_unet = self.netG(self.real_A)  # G(A)
            if self.opt.generate_residue:
                residue = torch.zeros(self.post_unet.shape).to(self.device)
                residue[:, out_h, :, :] = self.real_A[:, in_h, :, :] if not self.opt.linear else self.residue
                self.post_unet = self.post_unet + residue
            if self.opt.preload_unet:
                if self.opt.temp_unet_fix: self.post_unet[:, out_f, :, :] = self.post_unet[:, out_f, :, :] * 5 - 4
                self.post_unet = self.post_unet.detach()
        if self.opt.use_erosion and self.opt.erosion_only and self.opt.erosion_flowmap:
            terrain, water = self.netErosion(self.real_A[:, in_h, :, :], self.real_A[:, in_h, :, :])
            self.fake_B = torch.cat((terrain.float(), water.float().unsqueeze(1)), 1)
        elif self.opt.use_erosion and not self.opt.erosion_only:
            mult = (1 / 824 * 2) if self.opt.linear else 1
            bias = -1 if self.opt.linear else 0
            self.fake_B = self.post_unet.clone()
            self.fake_B[:, out_h, :, :] = self.netErosion(self.post_unet[:, out_h, :, :] * mult + bias, self.real_A[:, in_h, :, :]).float().squeeze(1)  # G(A)
        elif self.opt.use_erosion:
            iterations = None if not self.opt.debug_gradients else self.epoch
            self.fake_B = self.netErosion(self.real_A[:, in_h, :, :], self.real_A[:, in_h, :, :], iterations=iterations, store_water=self.opt.store_water)
            if self.opt.store_water:
                water = self.fake_B[1]
                self.fake_B = self.fake_B[0]
                for i in range(len(water)):
                    exrlib.write_exr('temp' + str(i) + '.exr', water[i].detach().cpu().float().numpy().transpose([1, 2, 0]), [str(i) for i in range(water[i].shape[0])])
            temp = self.real_B.clone()
            temp[:, out_h, :, :] = self.fake_B.float().squeeze(1)
            self.fake_B = temp
            
        if self.opt.exclude_flowmap and not self.isTrain:
            self.post_unet = torch.cat((torch.zeros_like(self.post_unet), self.post_unet), 1)
            self.real_B = torch.cat((torch.zeros_like(self.real_B), self.real_B), 1)

    def backward_D(self):
        if self.opt.use_erosion:
            fake_B = self.fake_B
        else:
            fake_B = self.post_unet

        if self.opt.use_erosion or not self.opt.use_feature_extractor:
            self.loss_D_L2 = torch.zeros([1]).to(self.device)
            self.loss_D = self.loss_D_L2
            #self.loss_D = self.criterionL2(fake_B, self.real_B)
        else:
            fake_features = self.netFeature(fake_B.detach())
            real_features = self.netFeature(self.real_B)
            self.loss_D_L2 = -(self.opt.lambda_L2 * self.criterionL2(fake_features, real_features) + self.opt.lambda_L1 * self.criterionL1(fake_features, real_features))
            self.loss_D = self.loss_D_L2
            self.loss_D.backward()

    def backward_G(self):
        if self.opt.use_erosion:
            fake_B = self.fake_B
        else:
            fake_B = self.post_unet

        if self.opt.use_erosion:
            out_h = self.opt.output_height_channel
            out_f = self.opt.output_flow_channel
            self.loss_G_L2 = (self.opt.lambda_L2 * self.criterionL2(fake_B[:, out_h, :, :], self.real_B[:, out_h, :, :]) + self.opt.lambda_L1 * self.criterionL1(fake_B[:, out_h, :, :], self.real_B[:, out_h, :, :]))
            if self.opt.erosion_flowmap:
                self.loss_G_L2 = self.loss_G_L2 + (self.opt.lambda_L2 * self.criterionL2(fake_B[:, out_f, :, :], self.real_B[:, out_f, :, :]) + self.opt.lambda_L1 * self.criterionL1(fake_B[:, out_f, :, :], self.real_B[:, out_f, :, :]))
            self.loss_G = self.loss_G_L2
        elif not self.opt.use_feature_extractor:
            bias = 1 if not self.opt.linear else 86
            if self.opt.log_error:
                self.loss_G_L2 = (self.opt.lambda_L2 * self.criterionL2(torch.log(torch.nn.ReLU(True)(fake_B + bias)), torch.log(torch.nn.ReLU(True)(self.real_B + bias))) + self.opt.lambda_L1 * self.criterionL1(fake_B, self.real_B))
            else:
                self.loss_G_L2 = (self.opt.lambda_L2 * self.criterionL2(fake_B, self.real_B) + self.opt.lambda_L1 * self.criterionL1(fake_B, self.real_B))
            self.loss_G = self.loss_G_L2
        else:
            self.loss_G_L2 = (self.opt.lambda_L2 * self.criterionL2(fake_B, self.real_B) + self.opt.lambda_L1 * self.criterionL1(fake_B, self.real_B))
            fake_B_output = self.netFeature(fake_B)
            real_B_output = self.netFeature(self.real_B)
            feat_loss = self.opt.lambda_L2 * self.criterionL2(fake_B_output, real_B_output) + self.opt.lambda_L1 * self.criterionL1(fake_B_output, real_B_output)
            self.loss_G = self.loss_G_L2 + feat_loss / feat_loss.item() * self.loss_G_L2.item()
        self.loss_G.backward()
        if self.opt.use_erosion:
            torch.nn.utils.clip_grad_value_(self.netErosion.parameters(), 0.05)
            self.var_names, self.var_values, self.var_grads = self.netErosion.module.get_var_and_grad()

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
        if not self.opt.preload_unet: self.optimizer_G.zero_grad()
        if self.opt.use_erosion: self.optimizer_Erosion.zero_grad()
        self.backward_G()
        if not self.opt.preload_unet: self.optimizer_G.step()
        if self.opt.use_erosion: self.optimizer_Erosion.step()

    def compute_visuals(self, dataset=None):
        if not self.opt.fixed_example or dataset is None:
            return
        single = dataset.dataset.get_val_item(self.opt.fixed_index)
        self.real_A = single['A'].unsqueeze(0).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        self.real_B = single['B'].unsqueeze(0).to(self.device).repeat(len(self.gpu_ids), 1, 1, 1)
        if self.opt.exclude_flowmap: self.real_B = self.real_B[:, 1, :, :].unsqueeze(1)
        out_h = self.opt.output_height_channel if not self.opt.exclude_flowmap else 1
        if self.opt.linear:
            self.residue = single['A_orig'].unsqueeze(0)[:, self.opt.input_height_channel, :, :].to(self.device).repeat(len(self.gpu_ids), 1, 1)
            self.real_B[:, self.opt.output_height_channel, :, :] = single['B_orig'].unsqueeze(0)[:, out_h, :, :].to(self.device).repeat(len(self.gpu_ids), 1, 1)
        else:
            self.real_B = 1 - torch.nn.ReLU()(2 - torch.nn.ReLU()(self.real_B + 1)) #clip to [-1, 1]
        self.image_paths = [single['A_paths']]

        self.forward()
        #if self.opt.generate_residue:
        #    self.post_unet[:, 1, :, :] = 1 - torch.nn.ReLU()(2 - torch.nn.ReLU()(self.post_unet[:, 1, :, :] + 1))
        if self.opt.linear:
            self.post_unet[:, self.opt.output_height_channel, :, :] = (self.post_unet[:, self.opt.output_height_channel, :, :] - 412) / 996 * 2
            self.real_B[:, self.opt.output_height_channel, :, :] = (self.real_B[:, self.opt.output_height_channel, :, :] - 412) / 996 * 2
        if self.opt.break4:
            self.real_A = self.combine_from_4(self.real_A)
            self.real_B = self.combine_from_4(self.real_B)
            self.post_unet = self.combine_from_4(self.post_unet)
        if self.opt.exclude_flowmap:
            self.post_unet = torch.cat((torch.zeros_like(self.post_unet), self.post_unet), 1)
            self.real_B = torch.cat((torch.zeros_like(self.real_B), self.real_B), 1)

    def update_epoch_params(self, epoch):
        self.epoch = epoch
        if self.opt.half_lr:
            lr = self.opt.lr * (2 ** (-epoch // 20))
            for optimizer in self.optimizers:
                state_dict = optimizer.state_dict()
                for param_group in state_dict['param_groups']:
                    param_group['lr'] = lr

    def get_current_visuals(self):
        if not self.opt.exclude_input:
            return super().get_current_visuals()
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[""] = getattr(self, name)
        return visual_ret

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
