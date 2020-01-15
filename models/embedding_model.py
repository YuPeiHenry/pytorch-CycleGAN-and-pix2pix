import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
import numpy as np


class EmbeddingModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', norm_G='instance', netG='gata', dataset_mode='exr_one_channel', input_nc=1, output_nc=1, preprocess='N.A.', image_type='exr', no_flip=True)
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        parser.add_argument('--max_filters', type=int, default=512, help='')
        parser.add_argument('--depth', type=int, default=9, help='')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'A_e', 'B_e']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_A_e', 'fake_B_e']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method, max_filters=opt.max_filters, depth=opt.depth)


        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.normalized_A = input['normalized_A'].to(self.device)
        self.normalized_B = input['normalized_B'].to(self.device)
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        self.forward_A_e()
        self.fake_A_e.detach()
        self.forward_B_e()
        self.fake_B_e.detach()
        #self.forward_A_i()
        #self.fake_A_i.detach()
        #self.forward_B_i()
        #self.fake_B_i.detach()

    def forward_A_e(self):
        self.fake_A_e = self.netG(self.normalized_B, 'un_erosion') + self.real_B

    def forward_B_e(self):
        self.fake_B_e = self.netG(self.normalized_A, 'erosion') + self.real_A

    def forward_A_i(self):
        self.fake_A_i = self.netG(self.normalized_A, 'zero_erosion') + self.real_A

    def forward_B_i(self):
        self.fake_B_i = self.netG(self.normalized_B, 'zero_erosion') + self.real_B

    def backward_D(self):
        self.loss_D = torch.zeros([1]).to(self.device)

    def backward_G(self):
        self.forward_A_e()
        self.loss_A_e = self.criterionL2(self.fake_A_e, self.real_A)
        #self.forward_A_i()
        #self.loss_A_i = self.criterionL2(self.fake_A_i, self.real_A)
        self.forward_B_e()
        self.loss_B_e = self.criterionL2(self.fake_B_e, self.real_B)
        #self.forward_B_i()
        #self.loss_B_i = self.criterionL2(self.fake_B_i, self.real_B)

        self.optimizer_G.zero_grad()
        (self.loss_B_e + self.loss_A_e).backward()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.backward_D()
        self.backward_G()
        self.loss_G = self.loss_B_e + self.loss_A_e

    def compute_visuals(self, dataset=None):
        if not self.opt.fixed_example or dataset is None:
            return
        single = dataset.dataset.get_val_item(self.opt.fixed_index)
        self.normalized_A = single['normalized_A'].unsqueeze(0).to(self.device)
        self.real_A = single['A'].unsqueeze(0).to(self.device)
        self.real_B = single['B'].unsqueeze(0).to(self.device)
        self.image_paths = [single['A_paths']]

        self.netG.eval()
        self.forward()
        self.netG.train()
