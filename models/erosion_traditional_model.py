import torch
from .base_model import BaseModel
from . import networks
import numpy as np


class ErosionTraditionalModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='erosion', lr=0.0002, input_nc=3, output_nc=1, preprocess='N.A.', image_type='uint16', image_value_bound=26350, no_flip=True)
        parser.add_argument('--width', type=int, default=512)
        parser.add_argument('--iterations', type=int, default=10)
        parser.add_argument('--debug', type=int, default=0)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        # define networks
        self.netG = networks.init_net(networks.ErosionLayer(opt.width, opt.iterations), gpu_ids=[self.device])
        self.z1 = np.random.rand(1, self.opt.iterations, self.opt.width, self.opt.width)
        self.z1 = torch.autograd.Variable(torch.from_numpy(self.z1), requires_grad=False).to(self.device)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        batch_size = self.real_A.size()[0]
        z2 = np.random.rand(batch_size, self.opt.iterations, self.opt.width, self.opt.width)
        z2 = torch.autograd.Variable(torch.from_numpy(z2), requires_grad=False).to(self.device)
        noise2 = z2
        self.fake_B = self.netG(self.real_A[:, 1, :, :].unsqueeze(1), z2)  # G(A)

    def backward_D(self):
        self.loss_D = torch.zeros([1]).to(self.device)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B.double()) * 10000
        # combine loss and calculate gradients
        if self.opt.debug == 1:
            for name, p in self.netG.named_parameters():
                if p.grad is None:
                    continue
                print(name)
                print(p.data)
                print(p.grad.data)
        self.loss_G = self.loss_G_L2
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        self.backward_D()
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

