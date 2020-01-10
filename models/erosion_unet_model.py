import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os
import data.exrlib as exrlib

class ErosionUnetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', norm_G='instance', netG='unet_256', dataset_mode='exr', input_nc=3, output_nc=2, preprocess='N.A.', image_type='exr', no_flip=True)
        parser.add_argument('--downsampleConv', type=int, default=0)
        parser.add_argument('--upsampleConv', type=int, default=0)
        parser.add_argument('--maxFilters', type=int, default=512)

        parser.add_argument('--SGD', action='store_true', help='')
        parser.add_argument('--input_height_channel', type=int, default=0)
        parser.add_argument('--output_height_channel', type=int, default=1)
        parser.add_argument('--output_flow_channel', type=int, default=0)
        parser.add_argument('--exclude_input', action='store_true', help='')
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        parser.add_argument('--width', type=int, default=512)
        parser.add_argument('--iterations', type=int, default=10)
        parser.add_argument('--store_water', action='store_true', help='')
        parser.add_argument('--lambda_L1', type=float, default=0.0, help='')
        parser.add_argument('--lambda_L2', type=float, default=1.0, help='')
        parser.add_argument('--erosion_lr', type=float, default=0.0001, help='')
        parser.add_argument('--use_feature_extractor', action='store_true', help='')
        parser.add_argument('--disable_cudnn', action='store_true', help='')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        if opt.disable_cudnn: torch.backends.cudnn.enabled = False
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'E', 'G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B'] if not opt.exclude_input else []
        self.visual_names += ['fake_B', 'post_erosion']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G', 'Erosion']
        if opt.use_feature_extractor: self.model_names += ['Feature']
        # define networks
        self.netG = networks.define_G(opt.input_nc + 1, opt.output_nc, opt.ngf, opt.netG, opt.maxFilters, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, linear=True, numDownsampleConv=opt.downsampleConv, numUpsampleConv=opt.upsampleConv)
        self.netErosion = networks.init_net(networks.ErosionLayer(opt.width, opt.iterations), gpu_ids=self.gpu_ids)
        if opt.use_feature_extractor:
            self.netFeature = networks.init_net(networks.FeatureExtractor(opt.output_nc), gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        if self.isTrain and opt.SGD:
            self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9)
            self.optimizers.append(self.optimizer_G)
            self.optimizer_Erosion = torch.optim.SGD(self.netErosion.parameters(), lr=opt.erosion_lr, momentum=0.9)
            self.optimizers.append(self.optimizer_Erosion)
            if opt.use_feature_extractor:
                self.optimizer_Feature = torch.optim.SGD(self.netFeature.parameters(), lr=opt.lr, momentum=0.9)
                self.optimizers.append(self.optimizer_Feature)
        elif self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            opt.preload_unet: self.optimizers.append(self.optimizer_G)
            self.optimizer_Erosion = torch.optim.Adam(self.netErosion.parameters(), lr=opt.erosion_lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_Erosion)
            if opt.use_feature_extractor:
                self.optimizer_Feature = torch.optim.Adam(self.netFeature.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_Feature)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.erosion_B = self.real_B[:, self.opt.output_height_channel, :, :].unsqueeze(1)
        self.residue = input['A_orig'][:, self.opt.input_height_channel, :, :].to(self.device)
        self.real_B[:, self.opt.output_height_channel, :, :] = input['B_orig'][:, self.opt.output_height_channel, :, :].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        in_h = self.opt.input_height_channel
        out_h = self.opt.output_height_channel
        out_f = self.opt.output_flow_channel

        self.post_erosion = self.netErosion(self.real_A[:, in_h, :, :], self.real_A[:, in_h, :, :], iterations=self.opt.iterations).float()
        self.fake_B = self.netG(torch.cat((self.real_A, (self.post_erosion - self.real_A[:, in_h, :, :].unsqueeze(1)).detach() * 10), 1))
        residue = torch.zeros(self.fake_B.shape).to(self.device)
        residue[:, out_h, :, :] = self.residue
        self.fake_B = self.fake_B + residue

    def backward_D(self):
        if not self.opt.use_feature_extractor:
            self.loss_D = torch.zeros([1]).to(self.device)
            #self.loss_D = self.criterionL2(fake_B, self.real_B)
        else:
            fake_features = self.netFeature(self.fake_B.detach())
            real_features = self.netFeature(self.real_B)
            self.loss_D = -(self.opt.lambda_L2 * self.criterionL2(fake_features, real_features) + self.opt.lambda_L1 * self.criterionL1(fake_features, real_features))
            self.loss_D.backward()

    def backward_E(self):
        out_h = self.opt.output_height_channel
        out_f = self.opt.output_flow_channel
        self.loss_E = (self.opt.lambda_L2 * self.criterionL2(self.post_erosion, self.erosion_B) + self.opt.lambda_L1 * self.criterionL1(self.post_erosion, self.erosion_B))
        self.loss_E.backward()

        torch.nn.utils.clip_grad_value_(self.netErosion.parameters(), 0.05)
        self.var_names, self.var_values, self.var_grads = self.netErosion.module.get_var_and_grad()

    def backward_G(self):
        self.loss_G = (self.opt.lambda_L2 * self.criterionL2(self.fake_B, self.real_B) + self.opt.lambda_L1 * self.criterionL1(self.fake_B, self.real_B))

        if self.opt.use_feature_extractor:
            fake_B_output = self.netFeature(self.fake_B)
            real_B_output = self.netFeature(self.real_B)
            feat_loss = self.opt.lambda_L2 * self.criterionL2(fake_B_output, real_B_output) + self.opt.lambda_L1 * self.criterionL1(fake_B_output, real_B_output)
            self.loss_G = self.loss_G + feat_loss / feat_loss.item() * self.loss_G.item()
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
        self.optimizer_Erosion.zero_grad()
        self.backward_E()
        self.optimizer_Erosion.step()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def compute_visuals(self, dataset=None):
        if not self.opt.fixed_example or dataset is None:
            return
        single = dataset.dataset.get_val_item(self.opt.fixed_index)
        self.real_A = single['A'].unsqueeze(0).to(self.device)
        self.real_B = single['B'].unsqueeze(0).to(self.device)
        self.residue = single['A_orig'].unsqueeze(0)[:, self.opt.input_height_channel, :, :].to(self.device)
        self.real_B[:, self.opt.output_height_channel, :, :] = single['B_orig'].unsqueeze(0)[:, self.opt.output_height_channel, :, :].to(self.device)
        self.image_paths = [single['A_paths']]

        self.forward()
        #if self.opt.generate_residue:
        #    self.post_unet[:, 1, :, :] = 1 - torch.nn.ReLU()(2 - torch.nn.ReLU()(self.post_unet[:, 1, :, :] + 1))
        self.post_unet[:, self.opt.output_height_channel, :, :] = self.post_unet[:, self.opt.output_height_channel, :, :] / 824 * 2 - 1

    def update_epoch_params(self, epoch):
        self.epoch = epoch
