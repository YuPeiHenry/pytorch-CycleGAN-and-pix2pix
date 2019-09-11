import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class Pix2PixHDModel(BaseModel):
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='instance')

        parser.add_argument('--relativistic', type=int, default=0, help='relativistic loss')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='feature matching loss')
        parser.add_argument('--use_instance', action='store_true')
        parser.add_argument('--use_gan_feat_loss', action='store_true', help='feature matching loss')
        parser.add_argument('--use_vgg_loss', action='store_true', help='vgg loss')
        parser.add_argument('--n_downsample_global', type=int, default=3, help='')
        parser.add_argument('--n_blocks_global', type=int, default=9, help='')
        parser.add_argument('--n_local_enhancers', type=int, default=1, help='')
        parser.add_argument('--n_blocks_local', type=int, default=3, help='')
        parser.add_argument('--num_D', type=int, default=3, help='')
        parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
        parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')        
        
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.use_features = opt.use_instance
        
        self.use_gan_feat_loss = opt.use_gan_feat_loss
        self.use_vgg_loss = opt.use_vgg_loss
        self.loss_names = ['G_GAN', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.use_gan_feat_loss:
            self.loss_names.append('G_GAN_Feat')
        if self.use_vgg_loss:
            self.loss_names.append(['G_VGG'])

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        """if self.use_features:
            self.model_names.append('E')"""
        if self.isTrain:
            self.model_names.append('D')

        ##### define networks
        # Generator network
        netG_input_nc = opt.input_nc        
        if opt.use_instance:
            netG_input_nc += 1
        """if self.use_features:
            netG_input_nc += opt.feat_num"""
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, self.gpu_ids,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local)

        # Discriminator network
        if self.isTrain:
            netD_input_nc = opt.input_nc + opt.output_nc
            if opt.use_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                          opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, opt.num_D, self.use_gan_feat_loss, not opt.gan_mode=='lsgan')

        """### Encoder network
        if self.use_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', opt.norm, not opt.no_dropout,
                                          opt.init_type, opt.init_gain, self.gpu_ids,
                                          opt.n_downsample_E)"""

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANHDLoss(use_lsgan=opt.gan_mode=='lsgan').to(self.device)
            self.criterionFeat = torch.nn.L1Loss()
            if self.use_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            """if self.use_features:              
                params += list(self.netE.parameters())"""
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def encode_input(self, infer=False):
        # get edges from instance map
        if self.use_features:
            edge_map = self.get_edges(self.inst)
            self.input = torch.cat((self.real_A, edge_map), dim=1)
        else:
            self.input = self.real_A

    def discriminate(self, input, test_image, use_pool=False):
        input_concat = torch.cat((input, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, infer=False):
        # Encode Inputs
        self.encode_input(infer=infer)  

        # Fake Generation
        """if self.use_features:
            feat_map = self.netE.forward(self.real_B, self.inst)                     
            input_concat = torch.cat((self.input, feat_map), dim=1)                        
        else:"""
        input_concat = self.input
        self.fake_B = self.netG.forward(input_concat)

    def list_avg(self, list_of_tensors, average=True):
        values = []
        for elem in list_of_tensors:
            values += [torch.mean(elem[-1])]
        if average:
            return torch.mean(torch.stack(values))
        else:
            return torch.sum(torch.stack(values))

    def backward_D(self):
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(self.input, self.fake_B.detach(), use_pool=True)

        # Real Detection and Loss        
        self.pred_real = self.discriminate(self.input, self.real_B)
        
        if self.opt.relativistic != 0:
            self.loss_D_fake = self.criterionGAN(self.list_avg(pred_fake_pool, average=False) - self.list_avg(self.pred_real) + 1, False)
            self.loss_D_real = self.criterionGAN(self.list_avg(self.pred_real, average=False) - self.list_avg(pred_fake_pool) - 1, True)
        else:
            self.loss_D_fake = self.criterionGAN(pred_fake_pool, False)        
            self.loss_D_real = self.criterionGAN(self.pred_real, True)
        
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((self.input, self.fake_B), dim=1))        

        if self.opt.relativistic != 0:
            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.discriminate(self.input, self.real_B)
            self.loss_G_GAN = self.criterionGAN(self.list_avg(pred_fake) - self.list_avg(pred_real).detach() - 1, True)
        else:
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            
        # GAN feature matching loss
        self.loss_G_GAN_Feat = 0
        if self.use_gan_feat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    self.loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], self.pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        self.loss_G_VGG = 0
        if self.use_vgg_loss:
            self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * self.opt.lambda_feat
            
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_Feat + self.loss_G_VGG
        self.loss_G.backward()
        
    
    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        self.optimizers.remove(self.optimizer_G)
        """if self.use_features:
            params += list(self.netE.parameters())"""
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.inst = input['inst'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

"""
class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)
"""
        
