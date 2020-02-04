import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import spectral_norm
import functools
from torch.optim import lr_scheduler
from torch.autograd import Variable
from .layers import *
from .stylegan_modules import *
from .erosionlib import *
from .HRNet_modules import get_hrnet
import numpy as np
import types

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x

class pixelwise_norm_layer(nn.Module):
    def __init__(self, dummy=False):
        super(pixelwise_norm_layer, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5

def get_norm_layer(norm_type='instance', style_dim=512):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'adain':
        norm_layer = functools.partial(AdaptiveInstanceNorm, style_dim=style_dim)
    elif norm_type == 'pixel':
        norm_layer = pixelwise_norm_layer
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            #ignore adain layer
            if hasattr(m, 'bias') and m.bias is not None and m.bias.data[0] == 0 and m.bias.data[-1] == 1:
                return

            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[],
    n_downsample_global=0, n_blocks_global=0, n_local_enhancers=0, n_blocks_local=0, progressive=False, progressive_stages=4, downsample_mode='strided', upsample_mode='transConv', upsample_method='nearest', linear=False, numDownsampleConv=0, numUpsampleConv=0, max_filters=512, depth=6):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method)
    elif netG == 'noise_unet':
        net = NoiseUnetGenerator(input_nc, output_nc, 7, ngf, max_filters=max_filters, norm_layer=norm_layer, use_dropout=use_dropout, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method, linear=linear, numDownsampleConv=numDownsampleConv, numUpsampleConv=numUpsampleConv)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, max_filters=max_filters, norm_layer=norm_layer, use_dropout=use_dropout, styled=norm=='adain', progressive=progressive, n_stage=progressive_stages, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method, linear=linear, numDownsampleConv=numDownsampleConv, numUpsampleConv=numUpsampleConv)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, max_filters=max_filters, norm_layer=norm_layer, use_dropout=use_dropout, styled=norm=='adain', progressive=progressive, n_stage=progressive_stages, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method, linear=linear, numDownsampleConv=numDownsampleConv, numUpsampleConv=numUpsampleConv)
    elif netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 9, ngf, max_filters=max_filters, norm_layer=norm_layer, use_dropout=use_dropout, styled=norm=='adain', progressive=progressive, n_stage=progressive_stages, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method, linear=linear, numDownsampleConv=numDownsampleConv, numUpsampleConv=numUpsampleConv)
    elif netG == 'unet_resblock':
        net = UnetResBlock(input_nc, output_nc, ngf, depth=depth, inc_rate=2., activation=nn.ReLU(), 
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=True)
    elif netG == 'unet_encoder':
        net = UnetEncoder(input_nc, output_nc, ngf, depth=depth, num_no_skip=depth-1, inc_rate=2., activation=nn.ReLU(), 
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=True)
    elif netG == 'overfit':
        net = TestUnet(input_nc, output_nc)
    elif netG == 'hrnet':
        net = get_hrnet(input_nc, output_nc)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        net.init_weights
        return net
    elif netG == 'gata':
        net = GATAUnet(input_nc, output_nc, depth, ngf=ngf, max_filters=max_filters)
    elif netG == 'fixed':
        net = FixedUnet()
    elif netG == 'global':
        net = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':        
        net = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        net = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    elif netG == 'FCDenseNet57' or netG == 'FCDenseNet':
        net = FCDenseNet(
            in_channels=input_nc, down_blocks=(4, 4, 4, 4, 4),
            up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
            growth_rate=12, out_chans_first_conv=48, out_channels=output_nc, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method)
    elif netG == 'FCDenseNet67':
        net = FCDenseNet(
            in_channels=input_nc, down_blocks=(5, 5, 5, 5, 5),
            up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
            growth_rate=16, out_chans_first_conv=48, out_channels=output_nc, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method)
    elif netG == 'FCDenseNet103':
        net = FCDenseNet(
            in_channels=input_nc, down_blocks=(4,5,7,10,12),
            up_blocks=(12,10,7,5,4), bottleneck_layers=15,
            growth_rate=16, out_chans_first_conv=48, out_channels=output_nc, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    if not (netG == 'unet_256' or 'unet_128') and progressive:
        raise NotImplementedError('Generator only supports progressive unet_256!')
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], num_D=0, use_gan_feat_loss=False, use_sigmoid=False, progressive=False, progressive_stages=4, downsample_mode='strided', upsample_method='nearest'):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, progressive=progressive, n_stage=progressive_stages, downsample_mode=downsample_mode, upsample_method=upsample_method)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, progressive=progressive, n_stage=progressive_stages, downsample_mode=downsample_mode, upsample_method=upsample_method)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'multi_n_layers':
        net = MultiNLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, downsample_mode=downsample_mode, upsample_method=upsample_method)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    if not (netD == 'n_layers' or netD == 'basic') and progressive:
        raise NotImplementedError('Discriminator only supports progressive basic or n_layers!')
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.vgg.to(gpu_ids[0])
            self.vgg = torch.nn.DataParallel(self.vgg, gpu_ids)  # multi-GPUs
            
    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, relativistic=False):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        self.relativistic = relativistic
        if relativistic:
            if gan_mode == 'lsgan':
                return
            elif gan_mode == 'hinge':
                return
            else:
                raise NotImplementedError('relativistic gan mode %s not implemented' % gan_mode)

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, pred_fake=None, discriminator=False):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.relativistic:
            y = self.get_target_tensor(prediction, True)
            if self.gan_mode == 'lsgan':
                if discriminator:
                    return torch.mean((prediction - torch.mean(pred_fake) - y) ** 2)/2 , torch.mean((pred_fake - torch.mean(prediction) + y) ** 2)/2
                else:
                    return (torch.mean((prediction - torch.mean(pred_fake) + y) ** 2) + torch.mean((pred_fake - torch.mean(prediction) - y) ** 2))/2
            if self.gan_mode == 'hinge':
                if discriminator:
                    return torch.mean(torch.nn.ReLU()(1.0 - (prediction - torch.mean(pred_fake))))/2 , torch.mean(torch.nn.ReLU()(1.0 + (pred_fake - torch.mean(prediction))))/2
                else:
                    return (torch.mean(torch.nn.ReLU()(1.0 + (prediction - torch.mean(pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 - (pred_fake - torch.mean(prediction)))))/2

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', downsample_mode='strided', upsample_mode='transConv', upsample_method='nearest'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += getDownsample(ngf * mult, ngf * mult * 2, 3, 2, 1, use_bias, downsample_mode=downsample_mode) + [
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            upsample = getUpsample(ngf * mult, int(ngf * mult / 2), 3, 2, 1, use_bias, upsample_mode, upsample_method=upsample_method, output_padding=1)
            model += upsample + [norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout = False, use_bias = True):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, max_filters=512, norm_layer=nn.BatchNorm2d, use_dropout=False, styled=False, addNoise=False, progressive=False, n_stage=4, downsample_mode='strided', upsample_mode='transConv', upsample_method='nearest', linear=False, numDownsampleConv=0, numUpsampleConv=0, no_normalization=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        self.progressive = progressive
        self.n_stage = n_stage
        blocks = []
        
        if styled:
            norm_layer = get_norm_layer('adain', style_dim=8 * ngf)

        # construct unet structure
        outer_nc = min(max_filters, ngf * (2 ** (num_downs - 2)))
        inner_nc = min(max_filters, ngf * (2 ** (num_downs - 1)))
        # add the innermost layer
        unet_block = UnetSkipConnectionBlock(outer_nc, inner_nc, input_nc=None, submodule=None, norm_layer=norm_layer, styled=styled, innermost=True, progressive=progressive, addNoise=addNoise, downsample_mode=downsample_mode, upsample_mode=upsample_mode, numDownsampleConv=numDownsampleConv, numUpsampleConv=numUpsampleConv, no_normalization=no_normalization)
        
        for i in range(num_downs - 2):
            outer_nc = min(max_filters, ngf * (2 ** (num_downs - i - 3)))
            inner_nc = min(max_filters, ngf * (2 ** (num_downs - i - 2)))
            unet_block = UnetSkipConnectionBlock(outer_nc, inner_nc, input_nc=None, submodule=unet_block, norm_layer=norm_layer, styled=styled,  progressive=progressive, addNoise=addNoise, use_dropout=use_dropout and outer_nc >= max_filters, downsample_mode=downsample_mode, upsample_mode=upsample_mode, numDownsampleConv=numDownsampleConv, numUpsampleConv=numUpsampleConv, no_normalization=no_normalization)
            if (i >= num_downs - 2 - n_stage + 1):
                blocks.append(unet_block)
        unet_block = UnetSkipConnectionBlock(output_nc, min(max_filters, ngf), input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, styled=styled, progressive=progressive, addNoise=addNoise, downsample_mode=downsample_mode, upsample_mode=upsample_mode, linear=linear, numDownsampleConv=numDownsampleConv, numUpsampleConv=numUpsampleConv, no_normalization=no_normalization)  # add the outermost layer
        blocks.append(unet_block)
        self.model = unet_block
        if not self.progressive:
            return
            
        self.alpha = 1
        self.blocks = blocks
        self.current_block = 0
        self.complete = False
        self.from_rgb = [nn.Conv2d(input_nc, ngf * 2 ** (2 - i), kernel_size=1, stride=1, padding=0, bias=True) for i in range(self.n_stage - 1)]
        self.from_rgb.append(lambda x: x)
        self.to_rgb = [nn.Conv2d(ngf * (2 ** (self.n_stage - 1 - i)), output_nc, kernel_size=1, stride=1, padding=0, bias=True) for i in range(self.n_stage - 1)]
        self.to_rgb.append(lambda x: x)
        self.decimation = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_method)
        
        n = 0
        for layer in self.to_rgb + self.from_rgb:
            if not isinstance(layer, torch.nn.Module):
                continue
            setattr(self, 'layers' + str(n), layer)
            n += 1

    def forward(self, input):
        """Standard forward"""
        if not self.progressive or self.complete:
            if self.progressive or self.model.styled:
                return self.model(input)[0]
            return self.model(input)

        n = self.current_block
        factor = (self.n_stage - 1 - n)
        decimated_input = input
        for i in range(factor):
            decimated_input = self.decimation(decimated_input)
        if n == 0 or self.alpha >= 1:
            combined_output = self.to_rgb[n](self.blocks[n](self.from_rgb[n](decimated_input))[0])
            for i in range(factor):
                combined_output = self.upsample(combined_output)
            return combined_output        

        a = self.alpha
        decimated_input_rgb = self.from_rgb[n](decimated_input)
        further_decimated = self.decimation(decimated_input)
        further_decimated_rgb = self.from_rgb[n - 1](further_decimated)
        next_input = further_decimated_rgb * (1 - a) + self.blocks[n].down(decimated_input_rgb) * a
        
        output, style = self.blocks[n - 1](next_input)
        if n < self.n_stage - 1:
            upper_output = torch.cat([decimated_input_rgb, self.blocks[n].up_forward(output, style)], 1)
        else:
            upper_output = self.blocks[n].up_forward(output, style)
        combined_output = self.upsample(self.to_rgb[n - 1](output.clone()) * (1 - a)) + self.to_rgb[n](upper_output)* a
        
        for i in range(factor):
            combined_output = self.upsample(combined_output)
        return combined_output

    def update_alpha(self, alpha, current_block):
        self.alpha = alpha
        self.current_block = current_block
        self.complete = alpha >= 1 and current_block >= self.n_stage - 1
        
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, styled=False, addNoise=False, progressive=False, downsample_mode='strided', upsample_mode='transConv', upsample_method='nearest', linear=False, numDownsampleConv=0, numUpsampleConv=0, no_normalization=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.progressive = progressive
        self.styled = styled
        self.addNoise = addNoise
        if self.styled:
            use_bias = True
            if innermost:
                self.linear = nn.Sequential(*[nn.ReLU(True) if i % 2 == 0 else nn.Linear(inner_nc, inner_nc) for i in range(8)])
            else:
                self.adain = norm_layer(inner_nc)
                self.up_activation = nn.ReLU(True)

                if self.addNoise:
                    self.add_noise = NoiseInjection(inner_nc)
                    self.noise_transform = nn.Sequential(*([nn.ReLU(True) if i % 2 == 0 else nn.Conv2d(inner_nc, inner_nc, kernel_size=1, stride=1, padding=0) for i in range(7)] + [nn.Conv2d(inner_nc, 1, kernel_size=1, stride=1, padding=0)]))

        elif type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        if input_nc is None:
            input_nc = outer_nc
        
        self.input_transform = ResBlockUnet(input_nc, numDownsampleConv)
        concat_nc = input_nc + outer_nc if not outermost else outer_nc
        self.output_transform = ResBlockUnet(concat_nc, numUpsampleConv)
        self.out_conv = nn.Conv2d(concat_nc, outer_nc * 2 if not outermost else outer_nc, kernel_size=1, stride=1, padding=0)
        downconv = getDownsample(input_nc, inner_nc, 4, 2, 1, use_bias, downsample_mode=downsample_mode)
        downrelu = [nn.LeakyReLU(0.2, True)]
        downnorm = [norm_layer(inner_nc)] if not self.styled else [nn.InstanceNorm2d(inner_nc)]
        if no_normalization: downnorm = []
        uprelu = [nn.ReLU(True)] if not self.styled else []
        upnorm = [norm_layer(outer_nc)] if not self.styled else []
        if no_normalization: upnorm = []

        if outermost:
            upconv = getUpsample(inner_nc * 2, outer_nc, 4, 2, 1, True, upsample_mode, upsample_method=upsample_method)
            down = downconv + downrelu
            up = upconv + ([nn.Tanh()] if not linear else [])
            model = down + [submodule] + up
        elif innermost:
            upconv = getUpsample(inner_nc, outer_nc, 4, 2, 1, use_bias, upsample_mode, upsample_method=upsample_method)
            down = downconv + downrelu
            submodule = [DenseBlockUnet(inner_nc, numDownsampleConv)]
            up = upconv + upnorm + uprelu
            model = down + submodule + up
        else:
            upconv = getUpsample(inner_nc * 2, outer_nc, 4, 2, 1, use_bias, upsample_mode, upsample_method=upsample_method)
            down = downconv + downnorm + downrelu
            up = upconv + upnorm + uprelu

            if use_dropout:
                up = up + [nn.Dropout(0.5)]
            model = down + [submodule] + up

        if not self.progressive and not self.styled:
            self.model = nn.Sequential(*model)
            return
        
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.submodule = submodule

    def forward(self, x):
        if not self.progressive and not self.styled:
            x = self.input_transform(x)
            intermediate = self.model(x)
            if not self.outermost:
                intermediate = torch.cat([x, intermediate], 1)
            output = self.output_transform(intermediate)
            return self.out_conv(output)

        #TODO: Fix progressive/style code to use the above res/dense blocks
        style = None
        if self.submodule is None:
            intermediate = self.down(x)
            if self.styled:
                latent = intermediate.mean(-1).mean(-1).detach()
                style = self.linear(latent)
        else:
            intermediate, style = self.submodule(self.down(x))

        result = self.up_forward(intermediate, style)

        if not self.outermost:
            result = torch.cat([x, result], 1)

        return result, style

    def up_forward(self, intermediate, style=None):
        if self.styled and not self.submodule is None:
            noise, output = intermediate.chunk(2, 1)
            if self.addNoise:
                output = self.add_noise(output, self.noise_transform(noise))
            output = self.up_activation(output)
            output = self.adain(output, style)
            intermediate = torch.cat([noise, output], 1)

        return self.up(intermediate)

class ResBlockUnet(nn.Module):
    def __init__(self, in_c, num_conv):
        super(ResBlockUnet, self).__init__()
        self.in_c = in_c
        self.num_conv = num_conv
        self.relu = nn.ReLU(True)
        self.out_conv = nn.Conv2d(in_c * 2, in_c, kernel_size=1, stride=1, padding=0)
        for i in range(num_conv):
            setattr(self, 'conv' + str(i), nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        output = x
        for i in range(self.num_conv):
            output = self.relu(getattr(self, 'conv' + str(i))(output))
        return self.out_conv(torch.cat((output, x), 1))

class DenseBlockUnet(nn.Module):
    def __init__(self, in_c, num_conv):
        super(DenseBlockUnet, self).__init__()
        self.in_c = in_c
        self.num_conv = num_conv
        self.relu = nn.ReLU(True)
        self.out_conv = nn.Conv2d(in_c * (num_conv + 1), in_c, kernel_size=1, stride=1, padding=0)
        for i in range(num_conv):
            setattr(self, 'conv' + str(i), nn.Conv2d(in_c * (i + 1), in_c, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        outputs = [x]
        for i in range(self.num_conv):
            outputs.append(self.relu(getattr(self, 'conv' + str(i))(torch.cat(outputs, 1))))
        return self.out_conv(torch.cat(outputs, 1))

class NoiseUnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, max_filters=512, norm_layer=nn.BatchNorm2d, use_dropout=False, downsample_mode='strided', upsample_mode='transConv', upsample_method='nearest', linear=False, numDownsampleConv=0, numUpsampleConv=0, width=512):
        super(NoiseUnetGenerator, self).__init__()
        # construct unet structure
        outer_nc = min(max_filters, ngf * (2 ** (num_downs - 2)))
        inner_nc = min(max_filters, ngf * (2 ** (num_downs - 1)))
        # add the innermost layer
        unet_block = NoiseUnetSkipConnectionBlock(outer_nc, inner_nc, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, downsample_mode=downsample_mode, upsample_mode=upsample_mode, numDownsampleConv=numDownsampleConv, numUpsampleConv=numUpsampleConv, noise_width=int(width/(2 ** (num_downs - 1))))
        
        for i in range(num_downs - 2):
            outer_nc = min(max_filters, ngf * (2 ** (num_downs - i - 3)))
            inner_nc = min(max_filters, ngf * (2 ** (num_downs - i - 2)))
            unet_block = NoiseUnetSkipConnectionBlock(outer_nc, inner_nc, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout and outer_nc >= max_filters, downsample_mode=downsample_mode, upsample_mode=upsample_mode, numDownsampleConv=numDownsampleConv, numUpsampleConv=numUpsampleConv, noise_width=int(width/(2 ** (num_downs - i - 2))))
        unet_block = NoiseUnetSkipConnectionBlock(output_nc, min(max_filters, ngf), input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, downsample_mode=downsample_mode, upsample_mode=upsample_mode, linear=linear, numDownsampleConv=numDownsampleConv, numUpsampleConv=numUpsampleConv, noise_width=width)  # add the outermost layer
        self.model = unet_block

    def forward(self, input):
        return self.model(input)
        
class NoiseUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, downsample_mode='strided', upsample_mode='transConv', upsample_method='nearest', linear=False, numDownsampleConv=0, numUpsampleConv=0, noise_width=512):
        super(NoiseUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        if input_nc is None:
            input_nc = outer_nc
        
        self.input_transform = ResBlockUnet(input_nc, numDownsampleConv)
        concat_nc = input_nc + outer_nc if not outermost else outer_nc
        self.output_transform = ResBlockUnet(concat_nc, numUpsampleConv)
        self.out_conv = nn.Conv2d(concat_nc, outer_nc * 2 if not outermost else outer_nc, kernel_size=1, stride=1, padding=0)

        self.noise_weights = nn.Sequential(ResBlockUnet(concat_nc, numUpsampleConv), nn.Conv2d(concat_nc, 32, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.noise = torch.nn.Parameter(torch.cuda.FloatTensor(np.random.rand(1, 32, noise_width, noise_width)))
        self.noise.requires_grad = True
        self.noise_transform = nn.Conv2d(32, outer_nc * 2 if not outermost else outer_nc, kernel_size=1, stride=1, padding=0)

        downconv = getDownsample(input_nc, inner_nc, 4, 2, 1, use_bias, downsample_mode=downsample_mode)
        downrelu = [nn.LeakyReLU(0.2, True)]
        downnorm = [norm_layer(inner_nc)]
        uprelu = [nn.ReLU(True)]
        upnorm = [norm_layer(outer_nc)]

        if outermost:
            upconv = getUpsample(inner_nc * 2, outer_nc, 4, 2, 1, True, upsample_mode, upsample_method=upsample_method)
            down = downconv + downrelu
            up = upconv + ([nn.Tanh()] if not linear else [])
            model = down + [submodule] + up
        elif innermost:
            upconv = getUpsample(inner_nc, outer_nc, 4, 2, 1, use_bias, upsample_mode, upsample_method=upsample_method)
            down = downconv + downrelu
            submodule = [DenseBlockUnet(inner_nc, numDownsampleConv)]
            up = upconv + upnorm + uprelu
            model = down + submodule + up
        else:
            upconv = getUpsample(inner_nc * 2, outer_nc, 4, 2, 1, use_bias, upsample_mode, upsample_method=upsample_method)
            down = downconv + downnorm + downrelu
            up = upconv + upnorm + uprelu

            if use_dropout:
                up = up + [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.input_transform(x)
        intermediate = self.model(x)
        if not self.outermost:
            intermediate = torch.cat([x, intermediate], 1)
        output = self.output_transform(intermediate)

        noise_output = self.noise_weights(intermediate) * self.noise.repeat(batch_size, 1, 1, 1)
        return self.out_conv(output) + self.noise_transform(noise_output)

class GATAUnet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, max_filters=512):
        super(GATAUnet, self).__init__()
        # construct unet structure
        outer_nc = min(max_filters, ngf * (2 ** (num_downs - 2)))
        inner_nc = min(max_filters, ngf * (2 ** (num_downs - 1)))
        # add the innermost layer
        unet_block = GATASkipBlock(outer_nc, inner_nc, input_nc=None, submodule=None, innermost=True)
        
        for i in range(num_downs - 2):
            outer_nc = min(max_filters, ngf * (2 ** (num_downs - i - 3)))
            inner_nc = min(max_filters, ngf * (2 ** (num_downs - i - 2)))
            unet_block = GATASkipBlock(outer_nc, inner_nc, input_nc=None, submodule=unet_block)
        unet_block = GATASkipBlock(output_nc, min(max_filters, ngf), input_nc=input_nc, submodule=unet_block, outermost=True)  # add the outermost layer
        self.model = unet_block

    def forward(self, input, embedding='zero_erosion'):
        return self.model(input)
        
class GATASkipBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False):
        super(GATASkipBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if input_nc is None:
            input_nc = outer_nc
        
        concat_nc = input_nc + outer_nc if not outermost else outer_nc
        self.output_transform = ResBlockUnet(concat_nc, 1)
        self.out_conv = nn.Conv2d(concat_nc, outer_nc * 2 if not outermost else outer_nc, kernel_size=1, stride=1, padding=0)

        downconv = [spectral_norm(nn.Conv2d(input_nc, inner_nc, 4, 2, 1))]
        downrelu = [nn.LeakyReLU(0.2, True)]
        downnorm = [nn.BatchNorm2d(inner_nc)]
        uprelu = [nn.ReLU(True)]
        upnorm = [nn.BatchNorm2d(outer_nc)]

        if outermost:
            upconv = [nn.Upsample(scale_factor = 2, mode='bilinear'), nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(inner_nc * 2, outer_nc, 3, 1, 0))]
            self.down = nn.Sequential(*(downconv + downrelu))
            self.up = nn.Sequential(*(upconv))
            self.submodule = submodule
        elif innermost:
            upconv = [nn.Upsample(scale_factor = 2, mode='bilinear'), nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(inner_nc * 3, outer_nc, 3, 1, 0))]
            self.down = nn.Sequential(*(downconv + downrelu))
            self.up = nn.Sequential(*(upconv + upnorm + uprelu))
            self.submodule = GATAEmbedding(inner_nc * 2)
        else:
            upconv = [nn.Upsample(scale_factor = 2, mode='bilinear'), nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(inner_nc * 2, outer_nc, 3, 1, 0))]
            self.down = nn.Sequential(*(downconv + downnorm + downrelu))
            self.up = nn.Sequential(*(upconv + upnorm + uprelu))
            self.submodule = submodule

    def forward(self, x, embedding='zero_erosion'):
        intermediate = self.down(x)
        intermediate = self.submodule(intermediate, embedding)
        intermediate = self.up(intermediate)
        if not self.outermost:
            intermediate = torch.cat([x, intermediate], 1)
        output = self.output_transform(intermediate)

        return self.out_conv(output)

class GATAEmbedding(nn.Module):
    def __init__(self, nc):
        super(GATAEmbedding, self).__init__()
        self.erosion_embedding = torch.nn.Parameter(torch.cuda.FloatTensor(np.random.rand(1, nc, 1, 1)))
        self.erosion_embedding.requires_grad = True
        self.zero_erosion = torch.nn.Parameter(torch.cuda.FloatTensor(np.random.rand(1, nc, 1, 1)))
        self.zero_erosion.requires_grad = True
    def forward(self, x, embedding='zero_erosion'):
        batch_size = x.shape[0]
        if embedding == 'zero_erosion':
            emb = self.zero_erosion
        elif embedding == 'erosion':
            emb = self.erosion_embedding
        elif embedding == 'un_erosion':
            emb = self.zero_erosion - self.erosion_embedding
        return torch.cat((x, self.zero_erosion.repeat(batch_size, 1, 1, 1)), 1)

#Hardcoded for debugging
class FixedUnet(nn.Module):
    def __init__(self):
        super(FixedUnet, self).__init__()
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear')
        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2d_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2d_3 = nn.Conv2d(35, 64, kernel_size=3, stride=1, padding=1)
        self.conv2d_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2d_5 = nn.Conv2d(99, 128, kernel_size=3, stride=1, padding=1)
        self.conv2d_6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2d_7 = nn.Conv2d(227, 256, kernel_size=3, stride=1, padding=1)
        self.conv2d_8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2d_9 = nn.Conv2d(483, 512, kernel_size=3, stride=1, padding=1)
        self.conv2d_10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2d_11 = nn.Conv2d(995, 1024, kernel_size=3, stride=1, padding=1)
        self.conv2d_12 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv2d_13 = nn.Conv2d(2019, 2048, kernel_size=3, stride=1, padding=1)
        self.conv2d_14 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)
        self.conv2d_15 = nn.Sequential(nn.ReplicationPad2d((0, 1, 0, 1)),
            nn.Conv2d(4067, 1024, kernel_size=2, stride=1, padding=0))
        self.conv2d_16 = nn.Conv2d(3043, 1024, kernel_size=3, stride=1, padding=1)
        self.conv2d_17 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.conv2d_18 = nn.Sequential(nn.ReplicationPad2d((0, 1, 0, 1)),
            nn.Conv2d(4067, 512, kernel_size=2, stride=1, padding=0))
        self.conv2d_19 = nn.Conv2d(1507, 512, kernel_size=3, stride=1, padding=1)
        self.conv2d_20 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2d_21 = nn.Sequential(nn.ReplicationPad2d((0, 1, 0, 1)),
            nn.Conv2d(2019, 256, kernel_size=2, stride=1, padding=0))
        self.conv2d_22 = nn.Conv2d(739, 256, kernel_size=3, stride=1, padding=1)
        self.conv2d_23 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2d_24 = nn.Sequential(nn.ReplicationPad2d((0, 1, 0, 1)),
            nn.Conv2d(995, 128, kernel_size=2, stride=1, padding=0))
        self.conv2d_25 = nn.Conv2d(355, 128, kernel_size=3, stride=1, padding=1)
        self.conv2d_26 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2d_27 = nn.Sequential(nn.ReplicationPad2d((0, 1, 0, 1)),
            nn.Conv2d(483, 64, kernel_size=2, stride=1, padding=0))
        self.conv2d_28 = nn.Conv2d(163, 64, kernel_size=3, stride=1, padding=1)
        self.conv2d_29 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2d_30 = nn.Sequential(nn.ReplicationPad2d((0, 1, 0, 1)),
            nn.Conv2d(227, 32, kernel_size=2, stride=1, padding=0))
        self.conv2d_31 = nn.Conv2d(67, 32, kernel_size=3, stride=1, padding=1)
        self.conv2d_32 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2d_33 = nn.Conv2d(99, 2, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        input_1 = x
        conv2d_1 = self.relu(self.conv2d_1(input_1))
        conv2d_2 = self.relu(self.conv2d_2(conv2d_1))
        concat_1 = torch.cat((input_1, conv2d_2), 1)
        maxpool_1 = self.maxpool(concat_1)
        conv2d_3 = self.relu(self.conv2d_3(maxpool_1))
        conv2d_4 = self.relu(self.conv2d_4(conv2d_3))
        concat_2 = torch.cat((maxpool_1, conv2d_4), 1)
        maxpool_2 = self.maxpool(concat_2)
        conv2d_5 = self.relu(self.conv2d_5(maxpool_2))
        conv2d_6 = self.relu(self.conv2d_6(conv2d_5))
        concat_3 = torch.cat((maxpool_2, conv2d_6), 1)
        maxpool_3 = self.maxpool(concat_3)
        conv2d_7 = self.relu(self.conv2d_7(maxpool_3))
        conv2d_8 = self.relu(self.conv2d_8(conv2d_7))
        concat_4 = torch.cat((maxpool_3, conv2d_8), 1)
        maxpool_4 = self.maxpool(concat_4)
        conv2d_9 = self.relu(self.conv2d_9(maxpool_4))
        conv2d_10 = self.relu(self.conv2d_10(conv2d_9))
        concat_5 = torch.cat((maxpool_4, conv2d_10), 1)
        maxpool_5 = self.maxpool(concat_5)
        conv2d_11 = self.relu(self.conv2d_11(maxpool_5))
        conv2d_12 = self.relu(self.conv2d_12(conv2d_11))
        concat_6 = torch.cat((maxpool_5, conv2d_12), 1)
        maxpool_6 = self.maxpool(concat_6)
        conv2d_13 = self.relu(self.conv2d_13(maxpool_6))
        conv2d_14 = self.relu(self.conv2d_14(conv2d_13))
        concat_7 = torch.cat((maxpool_6, conv2d_14), 1)
        upsample_1 = self.upsample(concat_7)
        conv2d_15 = self.relu(self.conv2d_15(upsample_1))
        concat_8 = torch.cat((concat_6, conv2d_15), 1)
        conv2d_16 = self.relu(self.conv2d_16(concat_8))
        conv2d_17 = self.relu(self.conv2d_17(conv2d_16))
        concat_9 = torch.cat((concat_8, conv2d_17), 1)
        upsample_2 = self.upsample(concat_9)
        conv2d_18 = self.relu(self.conv2d_18(upsample_2))
        concat_10 = torch.cat((concat_5, conv2d_18), 1)
        conv2d_19 = self.relu(self.conv2d_19(concat_10))
        conv2d_20 = self.relu(self.conv2d_20(conv2d_19))
        concat_11 = torch.cat((concat_10, conv2d_20), 1)
        upsample_3 = self.upsample(concat_11)
        conv2d_21 = self.relu(self.conv2d_21(upsample_3))
        concat_12 = torch.cat((concat_4, conv2d_21), 1)
        conv2d_22 = self.relu(self.conv2d_22(concat_12))
        conv2d_23 = self.relu(self.conv2d_23(conv2d_22))
        concat_13 = torch.cat((concat_12, conv2d_23), 1)
        upsample_4 = self.upsample(concat_13)
        conv2d_24 = self.relu(self.conv2d_24(upsample_4))
        concat_14 = torch.cat((concat_3, conv2d_24), 1)
        conv2d_25 = self.relu(self.conv2d_25(concat_14))
        conv2d_26 = self.relu(self.conv2d_26(conv2d_25))
        concat_15 = torch.cat((concat_14, conv2d_26), 1)
        upsample_5 = self.upsample(concat_15)
        conv2d_27 = self.relu(self.conv2d_27(upsample_5))
        concat_16 = torch.cat((concat_2, conv2d_27), 1)
        conv2d_28 = self.relu(self.conv2d_28(concat_16))
        conv2d_29 = self.relu(self.conv2d_29(conv2d_28))
        concat_17 = torch.cat((concat_16, conv2d_29), 1)
        upsample_6 = self.upsample(concat_17)
        conv2d_30 = self.relu(self.conv2d_30(upsample_6))
        concat_18 = torch.cat((concat_1, conv2d_30), 1)
        conv2d_31 = self.relu(self.conv2d_31(concat_18))
        conv2d_32 = self.relu(self.conv2d_32(conv2d_31))
        concat_19 = torch.cat((concat_18, conv2d_32), 1)
        conv2d_33 = self.conv2d_33(concat_19)
        return conv2d_33

class UnetResBlock(nn.Module):
    def __init__(self, in_c = 3, out_ch=2, ngf=64, depth=4, inc_rate=2., activation=nn.ReLU(True), 
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
        super(UnetResBlock, self).__init__()
        self.model = LevelBlock(in_c, ngf, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)

        out_conv_channels = ngf if not residual else (ngf * 3 + in_c)
        self.out_conv = nn.Conv2d(out_conv_channels, out_ch, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.model(x)
        return self.out_conv(x)
    
class LevelBlock(nn.Module):
    def __init__(self, in_dim, dim, depth, inc, acti, do, bn, mp, up, res):
        super(LevelBlock, self).__init__()
        self.depth = depth
        inner_dim = int(inc * dim)
        post_conv1 = dim if not res else (dim + in_dim)
        if depth != 1:
            inner_post_conv2 = inner_dim if not res else (inner_dim * 3 + post_conv1)
        else:
            inner_post_conv2 = inner_dim if not res else (inner_dim + post_conv1)
        pre_conv2 = dim + post_conv1
        self.conv1 = ConvBlock(in_dim, dim, acti, bn, res)
        
        if depth <= 0:
            return
        self.conv2 = ConvBlock(pre_conv2, dim, acti, bn, res)

        down = nn.MaxPool2d(2, 2) if mp else nn.Sequential(nn.Conv2d(post_conv1, post_conv1, kernel_size=3, stride=2, padding=1), acti)
        submodule = LevelBlock(post_conv1, inner_dim, depth - 1, inc, acti, do, bn, mp, up, res)
        up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.ReplicationPad2d((0, 1, 0, 1)), nn.Conv2d(inner_post_conv2 , dim, kernel_size=2, stride=1, padding=0), acti) if up else nn.Sequential(nn.ConvTranspose2d(inner_post_conv2, dim, kernel_size=3, stride=2, padding=1), acti)
        self.model = nn.Sequential(down, submodule, up)


    def forward(self, x):
        if self.depth <= 0:
            return self.conv1(x)
        n1 = self.conv1(x)
        m = self.model(n1)
        n2 = torch.cat((n1, m), 1)
        return self.conv2(n2)

class ConvBlock(nn.Module):
    def __init__(self, in_dim, dim, acti, bn, res, do=0):
        super(ConvBlock, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1), acti]
        if bn: layers += [nn.BatchNorm2d(dim)]
        if do: layers += [nn.Dropout(do)]
        layers += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1), acti]
        if bn: layers += [nn.BatchNorm2d(dim)]
        self.model = nn.Sequential(*layers)
        self.res = res
    def forward(self, x):
        output = self.model(x)
        if self.res: output = torch.cat((x, output), 1)
        return output

class UnetEncoder(nn.Module):
    def __init__(self, in_c = 3, out_ch=2, ngf=64, depth=4, num_no_skip=3, inc_rate=2., activation=nn.ReLU(True), 
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
        super(UnetEncoder, self).__init__()
        self.model = EncoderLevelBlock(in_c, ngf, depth, num_no_skip, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)

        out_conv_channels = ngf if not residual else (ngf * 2)
        self.out_conv = nn.Conv2d(out_conv_channels, out_ch, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.model(x)
        return self.out_conv(x)
    
class EncoderLevelBlock(nn.Module):
    def __init__(self, in_dim, dim, depth, num_no_skip, inc, acti, do, bn, mp, up, res):
        super(EncoderLevelBlock, self).__init__()
        self.depth = depth
        inner_dim = int(inc * dim)
        post_conv1 = dim if not res else (dim + in_dim)
        if num_no_skip != 1:
            inner_post_conv2 = inner_dim if not res else (inner_dim  * 2)
        else:
            inner_post_conv2 = inner_dim if not res else (inner_dim * 3 + post_conv1)
        pre_conv2 = dim
        self.conv1 = ConvBlock(in_dim, dim, acti, bn, res)
        
        if depth <= 0:
            return
        self.conv2 = ConvBlock(pre_conv2, dim, acti, bn, res)

        down = nn.MaxPool2d(2, 2) if mp else nn.Sequential(nn.Conv2d(post_conv1, post_conv1, kernel_size=3, stride=2, padding=1), acti)
        if num_no_skip > 1:
            submodule = EncoderLevelBlock(post_conv1, inner_dim, depth - 1, num_no_skip - 1, inc, acti, do, bn, mp, up, res)
        else:
            submodule = LevelBlock(post_conv1, inner_dim, depth - 1, inc, acti, do, bn, mp, up, res)
        up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.ReplicationPad2d((0, 1, 0, 1)), nn.Conv2d(inner_post_conv2 , dim, kernel_size=2, stride=1, padding=0), acti) if up else nn.Sequential(nn.ConvTranspose2d(inner_post_conv2, dim, kernel_size=3, stride=2, padding=1), acti)
        self.model = nn.Sequential(down, submodule, up)

    def forward(self, x):
        if self.depth <= 0:
            return self.conv1(x)
        n1 = self.conv1(x)
        m = self.model(n1)
        return self.conv2(m)

class TestUnet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(TestUnet, self).__init__()
        self.model = nn.Sequential(TestUnetSkip(input_nc, 32, 7, outermost=True), nn.Conv2d(64, output_nc, 1, 1, 0), nn.Tanh())

    def forward(self, x):
        return self.model(x)

class TestUnetSkip(nn.Module):
    def __init__(self, outer_nc, inner_nc, level, outermost=False):
        super(TestUnetSkip, self).__init__()
        self.outermost = outermost
        if not outermost: self.cross = TestResBlock(outer_nc, outer_nc)
        
        self.level = level
        if level <= 0:
            return
        conv1 = TestResBlock(outer_nc, inner_nc)
        down = nn.Sequential(nn.Conv2d(outer_nc + inner_nc, inner_nc, 4, 2, 1), nn.ReLU(True), nn.InstanceNorm2d(inner_nc))
        submodule = TestUnetSkip(inner_nc, inner_nc * 2, level - 1)
        up_nc = (outer_nc * 2) if not outermost else (inner_nc * 2)
        up = nn.Sequential(nn.Upsample(scale_factor = 2, mode='bilinear'), nn.ReflectionPad2d((0, 1, 0, 1)),
                nn.Conv2d(inner_nc * (3 if level != 1 else 2), up_nc, 2, 1, 0)
                , nn.ReLU(True), nn.InstanceNorm2d(up_nc))
        self.model = nn.Sequential(conv1, down, submodule, up)
        self.conv2 = TestResBlock(outer_nc * 2, outer_nc)
        
    def forward(self, x):
        if self.level <= 0:
            return self.cross(x)
        output = self.model(x)
        if self.outermost:
            return output
        cross = self.cross(x) + output
        return self.conv2(cross)

class TestResBlock(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(TestResBlock, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_nc, out_nc, 3, 1, 1), nn.ReLU(True), nn.InstanceNorm2d(out_nc), nn.Conv2d(out_nc, out_nc, 3, 1, 1), nn.ReLU(True), nn.InstanceNorm2d(out_nc))
    def forward(self, x):
        return torch.cat((x, self.model(x)), 1)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, progressive=False, n_stage=4, downsample_mode='strided', upsample_method='nearest'):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.progressive = progressive
        self.n_stage = n_stage

        kw = 4
        padw = 1
        blocks = [getDownsample(input_nc, ndf, kw, 2, padw, True, downsample_mode=downsample_mode) + [nn.LeakyReLU(0.2, True)]]
        sequence=[blocks[-1]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            blocks.append(getDownsample(nf_prev, nf, kw, 2, padw, True, downsample_mode=downsample_mode) +
                [norm_layer(nf), nn.LeakyReLU(0.2, True)])
            sequence.append(blocks[-1])

        nf_prev = nf
        nf = min(nf * 2, 512)
        blocks.append([
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf), nn.LeakyReLU(0.2, True)
        ])
        sequence.append(blocks[-1])

        blocks.append([nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)])

        if use_sigmoid:
            blocks[-1].append(nn.Sigmoid())
        sequence.append(blocks[-1])

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

        if not self.progressive:
            return
        sequence_stream = []
        for block in blocks[self.n_stage - 1:-1]:
            sequence_stream += block
        
        self.alpha = 1
        self.blocks = [nn.Sequential(*block) for block in blocks[0:self.n_stage - 1]] + [nn.Sequential(*sequence_stream)]
        self.current_block = 0
        self.complete = False
        self.from_rgb = [nn.Conv2d(input_nc, ndf * (2 ** i), kernel_size=1, stride=1, padding=0, bias=True) for i in range(3)]
        self.from_rgb = [lambda x: x] + self.from_rgb
        self.decimation = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_method)
        
        n = 0
        for layer in self.blocks + self.from_rgb:
            if not isinstance(layer, torch.nn.Module):
                continue
            setattr(self, 'layers' + str(n), layer)
            n += 1

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        elif not self.progressive:
            return self.model(input)
        elif self.complete:
            return self.forward_through(input, self.n_stage - 1)
        
        n = self.current_block
        factor = (self.n_stage - 1 - n)
        decimated_input = input
        for i in range(factor):
            decimated_input = self.decimation(decimated_input)
        if n == 0 or self.alpha >= 1:
            return self.forward_through(self.from_rgb[self.n_stage - 1 - n](decimated_input), n)
        
        a = self.alpha
        decimated_input = self.combine_mixed_res(decimated_input)
        further_decimated = self.decimation(decimated_input)
        further_decimated_rgb = self.from_rgb[self.n_stage - n](further_decimated)
        next_input = further_decimated_rgb * (1 - a) + self.blocks[self.n_stage - 1 - n](self.from_rgb[self.n_stage - 1 - n](decimated_input)) * a
        
        return self.forward_through(next_input, n - 1)
    
    def update_alpha(self, alpha, current_block):
        self.alpha = alpha
        self.current_block = current_block
        self.complete = alpha >= 1 and current_block >= self.n_stage - 1
        
    def forward_through(self, input, n):
        next_input = input
        for i in range(self.n_stage - 1 - n, self.n_stage):
            next_input = self.blocks[i](next_input)
        return next_input
    
    def combine_mixed_res(self, input):
        if self.alpha >= 1:
            return input
        return self.upsample(self.decimation(input)) * (1 - self.alpha) + input * self.alpha

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

#TODO: Add other forms of downsampling/upsampling
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

#TODO: Add other forms of downsampling/upsampling
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

class ErosionLayer(nn.Module):
    def __init__(self, width=512, iterations=10, output_water=False, random_param=False, blend_inputs=False, use_convs=False):
        super(ErosionLayer, self).__init__()
        self.width = width
        self.iterations = iterations
        self.output_water = output_water
        #self.blur = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.epsilon = 1e-10

        self.use_convs = use_convs
        if not use_convs:
            self.random_rainfall = torch.nn.Parameter(torch.cuda.DoubleTensor(np.random.rand(1, self.iterations, self.width, self.width)))
            self.random_rainfall.requires_grad = True
        else:
            self.in_rain_conv = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0)
            self.out_rain_conv = nn.Conv2d(32, iterations, kernel_size=1, stride=1, padding=0)
            for i in range(4):
                setattr(self, 'rain_conv' + str(i), nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(True)))
        self.random_gradient = torch.nn.Parameter(torch.cuda.DoubleTensor(np.random.rand(1, self.iterations, self.width, self.width)))
        self.random_gradient.requires_grad = False

        self.cell_width = 200 / self.width
        self.cell_area = self.cell_width ** 2
        # Learnable variables
        # Water-related constants
        
        self.alpha = torch.nn.Parameter(torch.cuda.DoubleTensor([0.0]))
        self.alpha.requires_grad = blend_inputs
        #inf
        if random_param:
            self.rain_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([np.random.uniform(-8, -2)]))
            self.evaporation_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([np.random.uniform(-12, -6)]))
            self.min_height_delta = torch.nn.Parameter(torch.cuda.DoubleTensor([np.random.uniform(-14, -8)]))
            self.gravity = torch.nn.Parameter(torch.cuda.DoubleTensor([np.random.uniform(2, 8)]))
            self.sediment_capacity_constant = torch.nn.Parameter(torch.cuda.DoubleTensor([np.random.uniform(1, 7)]))
            self.dissolving_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([np.random.uniform(-6, 0)]))
            self.deposition_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([np.random.uniform(-12, -6)]))
            self.max_height_delta = torch.nn.Parameter(torch.cuda.DoubleTensor([-8.965]))
        else:
            #self.rain_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([0.1 * self.cell_area]))
            self.rain_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([-6.0388]))
            #self.evaporation_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([0.02]))
            self.evaporation_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([-5.643]))
            #self.min_height_delta = torch.nn.Parameter(torch.cuda.DoubleTensor([0.0005]))
            self.min_height_delta = torch.nn.Parameter(torch.cuda.DoubleTensor([-10.965]))
            #self.gravity = torch.nn.Parameter(torch.cuda.DoubleTensor([30.0]))
            self.gravity = torch.nn.Parameter(torch.cuda.DoubleTensor([4.906]))
            #self.sediment_capacity_constant = torch.nn.Parameter(torch.cuda.DoubleTensor([50.0]))
            self.sediment_capacity_constant = torch.nn.Parameter(torch.cuda.DoubleTensor([5.643]))
            #self.dissolving_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([0.25]))
            self.dissolving_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([-2.0]))
            #self.deposition_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([0.05]))
            self.deposition_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([-4.321]))
            #self.max_height_delta = torch.nn.Parameter(torch.cuda.DoubleTensor([0.0020]))
            self.max_height_delta = torch.nn.Parameter(torch.cuda.DoubleTensor([-8.965]))

        self.rain_rate.requires_grad = True
        self.evaporation_rate.requires_grad = True
        self.min_height_delta.requires_grad = True
        self.gravity.requires_grad = True
        self.sediment_capacity_constant.requires_grad = True
        self.dissolving_rate.requires_grad = True
        self.deposition_rate.requires_grad = True
        self.max_height_delta.requires_grad = True
        
    def forward(self, input_terrain, original_terrain, iterations=None, store_water=False, init_water=None):
        if iterations is None:
            iterations = self.iterations
        iterations = min(iterations, self.iterations)
        if self.use_convs:
            intermediate = self.in_rain_conv(original_terrain.unsqueeze(1))
            for i in range(4):
                intermediate = getattr(self, 'rain_conv' + str(i))(intermediate) + intermediate
            self.random_rainfall = self.out_rain_conv(intermediate)

        coord_grid = np.array([[[[i, j] for i in range(self.width)] for j in range(self.width)]])
        self.coord_grid = torch.cuda.DoubleTensor(coord_grid).cuda()
        self.zeros = torch.cuda.DoubleTensor(np.zeros([1, self.width, self.width])).cuda()
        batch_size = input_terrain.size()[0]

        # These tensors are BatchSize x Height X Width
        terrain = ((1 + (self.alpha * input_terrain + (1 - self.alpha) * original_terrain)) / 2).view(-1, self.width, self.width).double()
        # `sediment` is the amount of suspended "dirt" in the water. Terrain will be
        # transfered to/from sediment depending on a number of different factors.
        sediment = self.zeros.clone().repeat(batch_size, 1, 1)
        # The amount of water. Responsible for carrying sediment.
        if init_water is None:
            water = sediment.clone()
        else:
            water = (init_water + 1) / 2
        # The water velocity.
        velocity = sediment.clone()

        water_history = []
        if store_water:
            water[:, 50:100, 200:250] = 10
        for i in range(0, iterations):
            # Add precipitation.
            if not store_water:
                water = water + self.relu((2 ** self.rain_rate) * self.random_rainfall[:, i])

            # Compute the normalized gradient of the terrain height to determine direction of water and sediment.
            # Gradient is 4D. BatchSize x Height X Width x 2
            gradient = simple_gradient(terrain, self.random_gradient[:, i].view(-1, self.width, self.width), self.epsilon)
            gradient = torch.cat((gradient[:, :, :, 1].unsqueeze(3), gradient[:, :, :, 0].unsqueeze(3)), 3)

            # Compute the difference between the current height the height offset by `gradient`.
            neighbor_height = sample(terrain, gradient, self.coord_grid, self.width)
            # NOTE: height_delta has approximately no gradient
            height_delta = terrain - neighbor_height

            # Update velocity
            #velocity_2 = torch.max(velocity ** 2 + (2 ** self.gravity.clone()) * height_delta, self.epsilon)
            e = 2.718281828459045
            e_8 = torch.cuda.DoubleTensor([e ** (-8)])
            max_term = self.epsilon
            velocity_2 = velocity ** 2 + (2 ** self.gravity.clone()) * height_delta
            velocity_2 = self.relu(velocity_2 - max_term) + torch.min(e_8, e ** (velocity_2 - max_term - 8)) + max_term
            velocity = torch.sqrt(velocity_2)

            # If the sediment exceeds the quantity, then it is deposited, otherwise terrain is eroded.
            #new_height_delta = torch.max(height_delta.clone(), self.min_height_delta.clone() / self.cell_width)
            max_term = (2 ** self.min_height_delta) / self.cell_width
            new_height_delta = self.relu(height_delta.clone() - max_term) + torch.min(e_8, e ** (height_delta - max_term - 8)) + max_term
            sediment_capacity = new_height_delta * velocity * water * 2 ** self.sediment_capacity_constant

            # Sediment is deposited as height is higher
            first_term_boolean = self.relu(torch.sign(-height_delta))
            #first_term = torch.min(self.relu(-height_delta), sediment)
            min_term = self.relu(-height_delta.clone())
            first_term = -self.relu(-sediment.clone() + min_term) + torch.max(-(e_8), -(e ** (-sediment + min_term - 8))) + min_term
            # Sediment is eroded as slope is too steep
            second_term_boolean = self.relu(torch.sign(height_delta - self.max_height_delta))
            second_term = second_term_boolean * (self.max_height_delta - height_delta)
            # Sediment is deposited as it exceeded capacity
            # Sediment is eroded otherwise
            sediment_diff = sediment - sediment_capacity
            third_term = (1 - first_term_boolean - second_term_boolean) * (self.relu(sediment_diff * 2 ** self.deposition_rate) - self.relu(-sediment_diff * 2 ** self.dissolving_rate))
            deposited_sediment = first_term + second_term + third_term

            # Don't erode more sediment than the current terrain height.
            #deposited_sediment = torch.max(-self.relu(height_delta), deposited_sediment)
            max_term = -self.relu(height_delta.clone())
            deposited_sediment = torch.relu(deposited_sediment - max_term) + torch.min(e_8, e ** (deposited_sediment - max_term - 8)) + max_term

            # Update terrain and sediment quantities.
            sediment = sediment - deposited_sediment
            terrain = terrain + deposited_sediment
            sediment = displace(sediment, gradient)
            water = displace(water, gradient)

            # Smooth out steep slopes.
            #terrain = self.apply_slippage(terrain, self.repose_slope, self.random_gradient[:, i].view(-1, self.width, self.width))

            # Update velocity
            #velocity = (2 ** self.gravity.clone()) * height_delta / self.cell_width
        
            # Apply evaporation
            water = water * (1 - 2 ** self.evaporation_rate.clone())
            
            if store_water:
                print(torch.mean(water))
                water_history.append(water.detach().clone())

        terrain = terrain.unsqueeze(1)
        terrain = terrain * 2 - 1
        if not store_water and not self.output_water:
            return terrain
        elif store_water:
            return (terrain, water_history)
        elif self.output_water:
            return (terrain, (water * 2) - 1)
        else:
            return terrain

    def get_var_and_grad(self):
        names = ['rain_rate', 'evaporation_rate', 'min_height_delta', 'gravity', 'sediment_capacity_constant', 'deposition_rate', 'dissolving_rate', 'max_height_delta']
        vars = [getattr(self, name).item() for name in names]
        grads = [getattr(self, name).grad.item() for name in names]
        return names, vars, grads

#Taken from https://github.com/bfortuner/pytorch_tiramisu

class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5,5,5,5,5),
                 up_blocks=(5,5,5,5,5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, out_channels=1, downsample_mode = 'strided', upsample_mode='transConv'):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count, downsample_mode=downsample_mode))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels, upsample_mode=upsample_mode))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels, upsample_mode=upsample_mode))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Output - used to be softmax ##
        self.tanh = nn.Tanh()
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=out_channels, kernel_size=1, stride=1,
                   padding=0, bias=True)

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.tanh(self.finalConv(out))
        return out

class StyledGenerator128(nn.Module):
    def __init__(self, input_nc, output_nc, n_class, task, ngf=64, norm_layer=nn.BatchNorm2d):
        super(StyledGenerator128, self).__init__()
        self.input_map1 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1),
            norm_layer(ngf))
        self.input_map2 = nn.Sequential(nn.Conv2d(input_nc + 1, ngf, kernel_size=3, stride=1, padding=1),
            norm_layer(ngf))
        self.task = task
        self.encoder = StyledEncoder128(n_class, task, ngf)
        self.decoder = StyledDecoder128(ngf)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        for i in range(5):
            setattr(self, 'feature_conv'+str(4 - i),
                nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2 ** i, output_nc, kernel_size=1, stride=1, padding=0)))
        self.tanh = nn.Tanh()

    def forward(self, input):
        if self.task == 'classify':
            return self.classify(input)
        elif self.task == 'embedding':
            return self.embedding(input)

    def classify(self, input):
        mapped_input = self.input_map1(input)
        logits = self.encoder(mapped_input)
        return logits
        
    def embedding(self, input):
        mapped_input = self.input_map2(input)
        encoder_outputs, embedding = self.encoder(mapped_input)
        encoder_outputs.reverse()
        decoder_outputs = self.decoder(encoder_outputs[0], embedding, encoder_outputs[1:] + [None])
        output = getattr(self, 'feature_conv'+str(0))(decoder_outputs[0])
        for i in range(1, 5):
            output = self.upsample(output) + getattr(self, 'feature_conv'+str(i))(decoder_outputs[i])
        return self.tanh(output)

class StyledDiscriminator(nn.Module):
    def __init__(self, input_nc, n_class, task, ngf=64, norm_layer=nn.BatchNorm2d):
        super(StyledDiscriminator, self).__init__()
        self.task = task
        self.input_map = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1),
            norm_layer(ngf))
        self.encoder = StyledEncoder128(n_class, task, ngf)

    def forward(self, input):
        if self.task == 'classify':
            return self.classify(input)
        elif self.task == 'embedding':
            return self.embedding(input)
        elif self.task == 'discriminate':
            return self.discriminate(input)


    def classify(self, input):
        mapped_input = self.input_map(input)
        logits = self.encoder(mapped_input)
        return logits
        
    def embedding(self, input):
        mapped_input = self.input_map(input)
        encoder_outputs, _ = self.encoder(mapped_input)
        return encoder_outputs

    def discriminate(self, input):
        mapped_input = self.input_map(input)
        return self.encoder(mapped_input)

class StyledEncoder128(nn.Module):
    def __init__(self, n_class, task, ngf):
        super(StyledEncoder128, self).__init__()
        self.task = task
        for i in range(5):
            setattr(self, 'blocks' + str(i), StyledEncoderBlock(ngf * 2 ** i))
        self.linear1 = nn.Sequential(nn.ReLU(True), nn.Linear(ngf * 2 ** 6 - ngf * 2, 1024))
        self.linear2 = nn.Sequential(nn.ReLU(True), nn.Linear(1024, 1024))
        self.linear3 = nn.Sequential(nn.ReLU(True), nn.Linear(1024, n_class))

    def forward(self, input):
        if self.task == 'classify':
            return self.classify(input)
        elif self.task == 'embedding':
            return self.embedding(input)
        elif self.task == 'discriminate':
            return self.discriminate(input)

    def classify(self, x):
        _, embedding = self.embedding(x)
        logits = self.linear3(embedding)
        return logits

    def embedding(self, x):
        outputs = [x]
        for i in range(5):
            outputs.append(getattr(self, 'blocks'+str(i))(outputs[-1]))
        outputs = outputs[1:]
        flat = torch.cat([output.mean(-1).mean(-1) for output in outputs], 1)
        embedding = self.linear2(self.linear1(flat))
        return outputs, embedding

    def discriminate(self, x):
        output = x
        for i in range(5):
            output = getattr(self, 'blocks'+str(i))(output)
        return output

class StyledEncoderBlock(nn.Module):
    def __init__(self, n_c, norm_layer=nn.BatchNorm2d):
        super(StyledEncoderBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(n_c, n_c, kernel_size=3, stride=1, padding=1), norm_layer(n_c))
        self.conv2 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(n_c, n_c, kernel_size=3, stride=1, padding=1), norm_layer(n_c))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
    
    def forward(self, x):
        feat1 = self.conv1(x.clone())
        feat2 = self.conv2(feat1.clone())
        concat = torch.cat((feat1, feat2), 1)
        return self.pool(concat + x.repeat(1, 2, 1, 1))

class StyledDecoder128(nn.Module):
    def __init__(self, ngf):
        super(StyledDecoder128, self).__init__()
        norm_layer = get_norm_layer('adain', style_dim=1024)
        setattr(self, 'blocks' + str(4), StyledDecoderBlock(ngf * 2 ** 0, norm_layer, noiseless=True))
        for i in range(1, 5):
            setattr(self, 'blocks' + str(4 - i), StyledDecoderBlock(ngf * 2 ** i, norm_layer))

    def forward(self, x, style, noise):
        outputs = [x]
        for i in range(5):
            outputs.append(getattr(self, 'blocks'+str(i))(outputs[-1], style, noise[i]))
        outputs = outputs[1:]
        return outputs

class StyledDecoderBlock(nn.Module):
    def __init__(self, output_nc, norm_layer, noiseless=False):
        super(StyledDecoderBlock, self).__init__()
        #upsample, conv, noise, activation, adain
        self.noiseless = noiseless
        self.upconv = nn.Sequential(*(getUpsample(output_nc * 2, output_nc, 3, 2, 1, True, 'upsample', upsample_method='nearest')))
        if not self.noiseless:
            self.add_noise = NoiseInjection(output_nc)
        self.uprelu = nn.ReLU(True)
        self.adain = norm_layer(output_nc)

    def forward(self, x, style, noise):
        output = self.upconv(x)
        if not self.noiseless:
            output = self.add_noise(output, noise)
        output = self.uprelu(output)
        output = self.adain(output, style)
        return output

class FeatureExtractor(nn.Module):
    def __init__(self, input_nc):
        super(FeatureExtractor, self).__init__()

        self.input_nc = input_nc
        self.conv2 = nn.Sequential(nn.ReplicationPad2d((0, 1, 0, 1)), nn.Conv2d(1, 12, kernel_size=2, stride=1, padding=0, bias=False))
        self.conv3 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_transform = nn.Sequential(nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False), nn.Tanh())

    def forward(self, input):
        return torch.cat([self.forward_single_channel(input[:, i].unsqueeze(1)) for i in range(self.input_nc)], 1)

    def forward_single_channel(self, input):
        features = torch.cat((self.conv2(input), self.conv3(input), self.conv5(input)), 1)
        return self.feat_transform(features)

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
