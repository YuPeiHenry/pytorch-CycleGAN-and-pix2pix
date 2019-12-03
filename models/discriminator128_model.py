import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os

class Discriminator128Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        #parser.set_defaults(norm='batch', norm_G='adain', netG='unet_128', netD='', dataset_mode='twocat', preprocess='resize', load_size=512, input_nc=1, output_nc=3, downsample_mode='downsample', upsample_mode='upsample', no_flip=True)
        parser.set_defaults(norm='batch', norm_G='batch', netG='unet_128', netD='n_layers', dataset_mode='twocat', preprocess='resize', load_size=512, input_nc=1, output_nc=3, upsample_mode='upsample', no_flip=True, gan_mode='lsgan')
        parser.set_defaults(save_epoch_freq=20, display_id=0, niter=200, niter_decay=0, save_latest_freq=481)
        parser.add_argument('--n_class', type=int, default=2, help='')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_128', 'D_512', 'D_full']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_B1', 'full_fake_B', 'full_real_B', 'predict_B1', 'predict_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.preload_names = ['U', 'G']
        self.model_names = ['D_128', 'D_512', 'D_full']
        # define networks
        self.netU = networks.define_G(1, opt.output_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method)
        self.netG = networks.StyledGenerator128(opt.output_nc, opt.output_nc, opt.n_class, 'embedding')
        self.netG = networks.init_net(self.netG, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_128 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_512 = networks.define_D(opt.input_nc + opt.output_nc * 2, opt.ndf * 2, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD_full = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.to128 = torch.nn.AvgPool2d(kernel_size=4, stride=4, padding=0, ceil_mode=False)
        self.from128 = torch.nn.Upsample(scale_factor=4, mode='nearest')

        if self.isTrain: self.load_base_networks()

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.optimizer_D_128 = torch.optim.Adam(self.netD_128.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_512 = torch.optim.Adam(self.netD_512.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_full = torch.optim.Adam(self.netD_full.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D_128)
            self.optimizers.append(self.optimizer_D_512)
            self.optimizers.append(self.optimizer_D_full)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.full_real_A = input['A' if AtoB else 'B'].to(self.device)
        self.full_real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_B1 = self.to128(self.full_real_B)
        self.full_fake_B = self.full_real_B.clone()
        self.real_A_128 = self.to128(self.full_real_A)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.opt.norm_G == 'adain':
            self.noise_inputs = []
            batch_size = self.full_real_A.size()[0]
            for length in self.netU.module.noise_length:
                z = (np.random.rand(batch_size, length, 1, 1).astype(np.float32) - 0.5) / 0.5
                z = torch.autograd.Variable(torch.from_numpy(z), requires_grad=False).to(self.device)
                self.noise_inputs.append(z)
        else:
            self.noise_inputs = None

    def forward(self):
        return

    def backward_D_128(self):
        fake_AB = torch.cat((self.real_A_128, self.fake_B1), 1)
        pred_fake = self.netD_128(fake_AB.detach())
        real_AB = torch.cat((self.real_A_128, self.real_B1), 1)
        pred_real = self.netD_128(real_AB)
        if self.isTrain:
            self.loss_D_128 = self.criterionGAN(pred_fake - torch.mean(pred_real) + 1, False) + self.criterionGAN(pred_real - torch.mean(pred_fake) - 1, True)
            self.loss_D_128.backward()
        self.predict_B1 = torch.cat((torch.ones([pred_fake.shape[0], 1, pred_fake.shape[2], pred_fake.shape[3]]), pred_fake.repeat(1, 2, 1, 1).cpu()), 1)

    def backward_D_512(self):
        fake_AB = torch.cat((self.real_A, self.real_B1, self.fake_B), 1)
        pred_fake = self.netD_512(fake_AB.detach())
        real_AB = torch.cat((self.real_A, self.real_B1, self.real_B), 1)
        pred_real = self.netD_512(real_AB)
        if self.isTrain:
            self.loss_D_512 = self.criterionGAN(pred_fake - torch.mean(pred_real) + 1, False) + self.criterionGAN(pred_real - torch.mean(pred_fake) - 1, True)
            self.loss_D_512.backward()

    def backward_D_full(self):
        fake_AB = torch.cat((self.full_real_A, self.full_fake_B), 1)
        pred_fake = self.netD_full(fake_AB.detach())
        real_AB = torch.cat((self.full_real_A, self.full_real_B), 1)
        pred_real = self.netD_full(real_AB)
        if self.isTrain:
            self.loss_D_full = self.criterionGAN(pred_fake - torch.mean(pred_real) + 1, False) + self.criterionGAN(pred_real - torch.mean(pred_fake) - 1, True)
            self.loss_D_full.backward()
        self.predict_B = torch.cat((torch.ones([pred_fake.shape[0], 1, pred_fake.shape[2], pred_fake.shape[3]]), pred_fake.repeat(1, 2, 1, 1).cpu()), 1)


    def backward_G(self):
        self.loss_G = torch.zeros([1]).to(self.device)

    def optimize_parameters(self):
        self.backward_G()
        self.optimizer_D_128.zero_grad()
        self.fake_B1 = self.netU(self.real_A_128, self.noise_inputs)
        self.backward_D_128()
        self.optimizer_D_128.step()

        self.cumulative_loss_D_512 = 0
        interval = self.opt.load_size // 4
        half_interval = interval // 2
        startX = np.random.randint(interval + 2) - half_interval
        startY = np.random.randint(interval + 2) - half_interval
        for i in range(3):
            for j in range(3):
                X = min(startX + interval * (i + 1), self.opt.load_size - interval)
                Y = min(startY + interval * (j + 1), self.opt.load_size - interval)
                self.fake_B1 = self.netU(self.real_A_128, self.noise_inputs)
                self.fake_B2 = self.from128(self.fake_B1)[:, :, X:X + interval, Y:Y + interval]
                self.real_A = self.full_real_A[:, :, X:X + interval, Y:Y + interval]
                self.real_B = self.full_real_B[:, :, X:X + interval, Y:Y + interval]
                self.run_optimizers()
                self.cumulative_loss_D_512 += self.loss_D_512.detach()
            
        for i in range(2):
            for j in range(2):
                X = min(startX + interval * (i + 1), self.opt.load_size - interval)
                Y = min(startY + interval * (j + 1), self.opt.load_size - interval)
                self.fake_B1 = self.netU(self.real_A_128, self.noise_inputs)
                self.fake_B2 = self.from128(self.fake_B1)[:, :, X:X + interval, Y:Y + interval]
                self.real_A = self.full_real_A[:, :, X:X + interval, Y:Y + interval]
                self.real_B = self.full_real_B[:, :, X:X + interval, Y:Y + interval]
                self.run_optimizers()
                self.cumulative_loss_D_512 += self.loss_D_512.detach()

        self.loss_D_512 = self.cumulative_loss_D_512 / 13

        self.compute_full_images()
        self.optimizer_D_full.zero_grad()
        self.backward_D_full()
        self.optimizer_D_full.step()
        self.loss_D = self.loss_D_128 + self.loss_D_512 + self.loss_D_full

    def run_optimizers(self):
        self.optimizer_D_512.zero_grad()
        self.fake_B = self.netG(torch.cat((self.real_A, self.fake_B2), 1))
        self.backward_D_512()
        self.optimizer_D_512.step()

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

    def load_base_networks(self, epoch='base'):
        for name in self.preload_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
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

    def compute_visuals(self, dataset=None):
        return

    def compute_full_images(self):
        self.fake_B1 = self.netU(self.real_A_128, self.noise_inputs).detach()
        self.fake_B1_ = self.from128(self.fake_B1)
        interval = self.opt.load_size // 4
        for i in range(4):
            for j in range(4):
                X = interval * i
                Y = interval * j
                self.fake_B2 = self.fake_B1_[:, :, X:X + interval, Y:Y + interval]
                self.real_A = self.full_real_A[:, :, X:X + interval, Y:Y + interval]
                self.fake_B = self.netG(torch.cat((self.real_A, self.fake_B2), 1)).detach()
                self.full_fake_B[:, :, X:X + interval, Y:Y + interval] = self.fake_B
