import torch
from .base_model import BaseModel
from .pix2pix_model import Pix2PixModel
from . import networks
import numpy as np


class ErosionModel(Pix2PixModel):
    def modify_commandline_options(parser, is_train=True):
        Pix2PixModel.modify_commandline_options(parser, is_train=is_train)
        parser.set_defaults(dataset_mode='erosion', input_nc=3, output_nc=1, preprocess='N.A.', image_type='uint16', image_value_bound=26350, no_flip=True)
        parser.add_argument('--fixed_example', action='store_true', help='')
        parser.add_argument('--fixed_index', type=int, default=0, help='')
        return parser

    def __init__(self, opt):
        Pix2PixModel.__init__(self, opt)

    def compute_visuals(self, dataset=None):
        if not self.opt.fixed_example or dataset is None:
            return
        single = dataset.__getitem__(self.opt.fixed_index)
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = single['A' if AtoB else 'B'].unsqueeze(0).to(self.device)
        self.real_B = single['B' if AtoB else 'A'].unsqueeze(0).to(self.device)
        self.image_paths = [single['A_paths' if AtoB else 'B_paths']]

        self.forward()
