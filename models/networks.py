import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.autograd import Variable
from .layers import *
from .stylegan_modules import *
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
    n_downsample_global=0, n_blocks_global=0, n_local_enhancers=0, n_blocks_local=0, progressive=False, progressive_stages=4, downsample_mode='strided', upsample_mode='transConv', upsample_method='nearest', linear=False):
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
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, styled=norm=='adain', progressive=progressive, n_stage=progressive_stages, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method, linear=linear)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, styled=norm=='adain', progressive=progressive, n_stage=progressive_stages, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method, linear=linear)
    elif netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout, styled=norm=='adain', progressive=progressive, n_stage=progressive_stages, downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method, linear=linear)
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
    elif netG == 'multi_unet':
        net = MultiUnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, styled=norm=='adain', downsample_mode=downsample_mode, upsample_mode=upsample_mode, upsample_method=upsample_method)
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

    if netD == 'multiscale':
        net = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, use_gan_feat_loss, downsample_mode=downsample_mode)
    elif netD == 'basic':  # default PatchGAN classifier
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

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, styled=False, addNoise=False, progressive=False, n_stage=4, downsample_mode='strided', upsample_mode='transConv', upsample_method='nearest', linear=False):
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
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, styled=styled, innermost=True, progressive=progressive, addNoise=addNoise, downsample_mode=downsample_mode, upsample_mode=upsample_mode)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, styled=styled,  progressive=progressive, addNoise=addNoise, use_dropout=use_dropout, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, styled=styled, progressive=progressive, addNoise=addNoise, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        blocks.append(unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, styled=styled, progressive=progressive, addNoise=addNoise, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        blocks.append(unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, styled=styled, progressive=progressive, addNoise=addNoise, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        blocks.append(unet_block)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, styled=styled, progressive=progressive, addNoise=addNoise, downsample_mode=downsample_mode, upsample_mode=upsample_mode, linear=linear)  # add the outermost layer
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
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, styled=False, addNoise=False, progressive=False, downsample_mode='strided', upsample_mode='transConv', upsample_method='nearest', linear=False):
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
        downconv = getDownsample(input_nc, inner_nc, 4, 2, 1, use_bias, downsample_mode=downsample_mode)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = [norm_layer(inner_nc)] if not self.styled else [nn.InstanceNorm2d(inner_nc)]
        uprelu = [nn.ReLU(True)] if not self.styled else []
        upnorm = [norm_layer(outer_nc)] if not self.styled else []

        if outermost:
            upconv = getUpsample(inner_nc * 2, outer_nc, 4, 2, 1, True, upsample_mode, upsample_method=upsample_method)
            down = downconv
            up = uprelu + upconv + ([nn.Tanh()] if not linear else [])
            model = down + [submodule] + up
        elif innermost:
            upconv = getUpsample(inner_nc, outer_nc, 4, 2, 1, use_bias, upsample_mode, upsample_method=upsample_method)
            down = [downrelu] + downconv
            up = uprelu + upconv + upnorm
            model = down + up
        else:
            upconv = getUpsample(inner_nc * 2, outer_nc, 4, 2, 1, use_bias, upsample_mode, upsample_method=upsample_method)
            down = [downrelu] + downconv + downnorm
            up = uprelu + upconv + upnorm

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
            if self.outermost:
                return self.model(x)
            return torch.cat([x, self.model(x)], 1)

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

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, downsample_mode=downsample_mode)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return model(input)

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class ErosionLayer(nn.Module):
    def __init__(self, width=512, iterations=10):
        super(ErosionLayer, self).__init__()
        self.width = width
        self.iterations = iterations
        #self.blur = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.epsilon = 1e-10

        self.random_rainfall = torch.nn.Parameter(torch.cuda.DoubleTensor(np.random.rand(1, self.iterations, self.width, self.width)))
        self.random_rainfall.requires_grad = False
        self.random_gradient = torch.nn.Parameter(torch.cuda.DoubleTensor(np.random.rand(1, self.iterations, self.width, self.width)))
        self.random_gradient.requires_grad = False

        self.cell_width = 200 / self.width
        self.cell_area = self.cell_width ** 2
        # Learnable variables
        # Water-related constants
        
        #inf
        self.rain_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([0.1 * self.cell_area]))
        self.rain_rate.requires_grad = True
        #inf
        self.evaporation_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([0.02]))
        self.evaporation_rate.requires_grad = True
        # Slope constants
        #inf
        self.min_height_delta = torch.nn.Parameter(torch.cuda.DoubleTensor([0.05]))
        self.min_height_delta.requires_grad = True
        self.height_epsilon = torch.nn.Parameter(torch.cuda.DoubleTensor([0.001]))
        self.height_epsilon.requires_grad = True
        #self.repose_slope = torch.nn.Parameter(torch.cuda.DoubleTensor([0.015]))
        #self.repose_slope.requires_grad = True
        #inf
        self.gravity = torch.nn.Parameter(torch.cuda.DoubleTensor([50.0]))
        self.gravity.requires_grad = True
        # Sediment constants
        #inf
        self.sediment_capacity_constant = torch.nn.Parameter(torch.cuda.DoubleTensor([15.0]))
        self.sediment_capacity_constant.requires_grad = True
        #inf
        self.dissolving_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([0.1]))
        self.dissolving_rate.requires_grad = True
        #0
        self.deposition_rate = torch.nn.Parameter(torch.cuda.DoubleTensor([0.0025]))
        self.deposition_rate.requires_grad = True
        
    def forward(self, input_terrain):
        coord_grid = np.array([[[[i, j] for i in range(self.width)] for j in range(self.width)]])
        self.coord_grid = torch.cuda.DoubleTensor(coord_grid).cuda()
        self.zeros = torch.cuda.DoubleTensor(np.zeros([1, self.width, self.width])).cuda()
        batch_size = input_terrain.size()[0]

        # These tensors are BatchSize x Height X Width
        terrain = ((1 - input_terrain) / 2).view(-1, self.width, self.width).double()
        # `sediment` is the amount of suspended "dirt" in the water. Terrain will be
        # transfered to/from sediment depending on a number of different factors.
        sediment = self.zeros.clone().repeat(batch_size, 1, 1)
        # The amount of water. Responsible for carrying sediment.
        water = sediment.clone()
        # The water velocity.
        velocity = sediment.clone()


        for i in range(0, self.iterations):
            # Add precipitation.
            water = water + self.relu(self.rain_rate.clone()) * self.random_rainfall[:, i]

            # Compute the normalized gradient of the terrain height to determine direction of water and sediment.
            # Gradient is 4D. BatchSize x Height X Width x 2
            gradient = self.simple_gradient(terrain, self.random_gradient[:, i].view(-1, self.width, self.width))

            # Compute the difference between the current height the height offset by `gradient`.
            neighbor_height = self.sample(terrain, -gradient)
            # NOTE: height_delta has approximately no gradient
            height_delta = terrain - neighbor_height

            # If the sediment exceeds the quantity, then it is deposited, otherwise terrain is eroded.
            height_delta_sign = torch.sign(self.relu(height_delta - self.height_epsilon))
            new_height_delta = height_delta_sign * torch.max(height_delta, self.min_height_delta.expand_as(height_delta))
            sediment_capacity = (new_height_delta / self.cell_width) * velocity * water * self.relu(self.sediment_capacity_constant.clone())

            # Sediment is deposited as height is higher
            first_term_boolean = self.relu(torch.sign(-height_delta))
            first_term = torch.min(self.relu(-height_delta), sediment)
            # Sediment is deposited as it exceeded capacity
            # Sediment is eroded otherwise
            sediment_diff = sediment - sediment_capacity
            third_term = (1 - first_term_boolean) * (self.relu(sediment_diff * self.deposition_rate) - self.relu(-sediment_diff * self.dissolving_rate))
            deposited_sediment = first_term + third_term

            # Don't erode more sediment than the current terrain height.
            deposited_sediment = torch.max(-self.relu(height_delta), deposited_sediment)

            # Update terrain and sediment quantities.
            sediment = sediment - deposited_sediment
            terrain = terrain + deposited_sediment
            sediment = self.displace(sediment, gradient)
            water = self.displace(water, gradient)

            # Smooth out steep slopes.
            #terrain = self.apply_slippage(terrain, self.repose_slope, self.random_gradient[:, i].view(-1, self.width, self.width))

            # Update velocity
            velocity = self.relu(self.gravity.clone()) * height_delta / self.cell_width
        
            # Apply evaporation
            water = water * (1 - self.relu(self.evaporation_rate.clone()))

        terrain = terrain.unsqueeze(1)
        return self.relu(1 + (1 - terrain * 2)) - 1

    def simple_gradient(self, input, noise):
        dx = 0.5 * torch.cat(((input[:, 0, :] * 1.1 - input[:, 0, :]).view(-1, 1, self.width),
            input[:, 2:, :] - input[:, :-2, :],
            (input[:, -1, :] * 0.9 - input[:,-1, :]).view(-1, 1, self.width)), 1)
        dy = 0.5 * torch.cat(((input[:, :, 0] * 1.1 - input[:, :, 0]).view(-1, self.width, 1),
            input[:, :, 2:] - input[:, :, :-2],
            (input[:, :, -1] * 0.9 - input[:, :, -1]).view(-1, self.width, 1)), 2)
        magnitude = torch.sqrt(dx * dx + dy * dy + self.epsilon / 10)

        randomX = noise
        randomY = torch.sqrt(1 - randomX * randomX)
        factor = self.relu(self.epsilon - magnitude)
        
        final_dx = (dx + factor * randomX) / (magnitude + factor)
        final_dy = (dy + factor * randomY) / (magnitude + factor)
        
        # 4D Tensor
        return torch.cat((final_dx.unsqueeze(3), final_dy.unsqueeze(3)), 3)

    def sample(self, input, offset):
        # coords are between [0, self.width - 1]. Normalize to [-1, 1]
        coords = self.coord_grid.repeat(input.size()[0], 1, 1, 1) + offset
        normalized = (coords / (self.width - 1) * 2) - 1
        # For example, values: x: -1, y: -1 is the left-top pixel of the input
        # values: x: 1, y: 1 is the right-bottom pixel of the input
        return nn.functional.grid_sample((input - 1).unsqueeze(1), normalized, mode='bilinear', padding_mode='zeros', align_corners=True).view(-1, self.width, self.width) + 1
 
    def displace(self, a, delta):
        """
        fns = {
            -1: lambda x: -x,
            0: lambda x: 1 - np.abs(x),
            1: lambda x: x,
        }"""
        delta_x, delta_y = delta[:, :, :, 0], delta[:, :, :, 1]
        delta_x = delta_x.unsqueeze(3)
        delta_y = delta_y.unsqueeze(3)
        # BatchSize x Height X Width x 3
        x_multipliers = self.relu(torch.cat((-delta_x, 1 - torch.abs(delta_x), delta_x), 3))
        y_multipliers = self.relu(torch.cat((-delta_y, 1 - torch.abs(delta_y), delta_y), 3))
        
        post_x = self.sum_3tensors_with_offsets(x_multipliers * a.unsqueeze(3).repeat(1, 1, 1, 3), 1)
        post_y = self.sum_3tensors_with_offsets(y_multipliers * a.unsqueeze(3).repeat(1, 1, 1, 3), 2)
        
        return post_y

    def sum_3tensors_with_offsets(self, tensors, offset_axis):
        tensor1, tensor2, tensor3 = tensors[:, :, :, 0], tensors[:, :, :, 1], tensors[:, :, :, 2]
        if offset_axis == 1:
            tensor1 = torch.cat((tensor1[:, :-1, :], (tensor1[:, 0, :] - tensor1[:, 0, :]).view(-1, 1, self.width)), 1)
            tensor3 = torch.cat(((tensor3[:, 0, :] - tensor3[:, 0, :]).view(-1, 1, self.width), tensor3[:, 1:, :]), 1)
        elif offset_axis == 2:
            tensor1 = torch.cat((tensor1[:, :, :-1], (tensor1[:, :, 0] - tensor1[:, :, 0]).view(-1, self.width, 1)), 2)
            tensor3 = torch.cat(((tensor3[:, :, 0] - tensor3[:, :, 0]).view(-1, self.width, 1), tensor3[:, :, 1:]), 2)
        return torch.sum(torch.cat((tensor1.unsqueeze(3), tensor2.unsqueeze(3), tensor3.unsqueeze(3)), 3), 3)

    def apply_slippage(self, terrain, repose_slope, noise):
        delta = self.simple_gradient(terrain, noise) / self.width
        smoothed = self.blur(terrain.unsqueeze(1).float()).view(-1, self.width, self.width).double()
        diff = torch.sqrt(delta[:, :, :, 0] ** 2 + delta[:, :, :, 1] ** 2) - self.repose_slope
        sign = self.relu(diff) / (torch.abs(diff) + self.epsilon)
        result = terrain * (1 - sign) + sign * smoothed
        return result

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

class MultiUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, styled=False, downsample_mode='strided', upsample_mode='transConv', upsample_method='nearest'):
        super(MultiUnetGenerator, self).__init__()
        self.num_downs = num_downs
        self.styled = styled
        core_norm_layer = norm_layer
        if styled:
            core_norm_layer = get_norm_layer('adain', style_dim=8 * ngf)
            norm_layer = nn.BatchNorm2d
        
        new_input_size = int(ngf / 2)
        # construct unet structure
        unet_block = ModifiedUnetBlock(ngf * 8, ngf * 8, new_input_size, submodule=None, norm_layer=core_norm_layer, styled=styled, downsample_mode=downsample_mode, upsample_mode=upsample_mode)  # add the innermost layer
        for i in range(num_downs - 4):          # add intermediate layers with ngf * 8 filters
            unet_block = ModifiedUnetBlock(ngf * 8, ngf * 8, new_input_size, submodule=unet_block, norm_layer=core_norm_layer, use_dropout=use_dropout, styled=styled, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = ModifiedUnetBlock(ngf * 4, ngf * 8, new_input_size, submodule=unet_block, norm_layer=core_norm_layer, styled=styled, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        unet_block = ModifiedUnetBlock(ngf * 2, ngf * 4, new_input_size, submodule=unet_block, norm_layer=core_norm_layer, styled=styled, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        unet_block = ModifiedUnetBlock(ngf, ngf * 2, new_input_size, submodule=unet_block, norm_layer=core_norm_layer, styled=styled,  downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.model = unet_block
        
        self.input_map = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(ngf), nn.LeakyReLU(0.2, True))
        self.input_map2 = nn.Sequential(nn.Conv2d(input_nc, new_input_size, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(new_input_size), nn.LeakyReLU(0.2, True))
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_method)
        for i in range(num_downs - 3, num_downs):
            exponent = min(num_downs - i - 1, 3)
            setattr(self, 'feature_conv'+str(i),
                nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2 * (2 ** exponent), ngf, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(ngf), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)))
        self.tanh = nn.Tanh()

    def forward(self, input):
        input1 = self.input_map(input)
        input2 = self.input_map2(input)
        num_downs = self.num_downs
        submodule_outputs =  self.model(input1, input2)[0]
        outputs = [getattr(self, 'feature_conv'+str(num_downs - 3))(submodule_outputs[num_downs - 3])]
        for i in range(num_downs - 3 + 1, self.num_downs):
            outputs.append(self.upsample(outputs[-1]) + getattr(self, 'feature_conv'+str(i))(submodule_outputs[i]))
        outputs.reverse() # Biggest output at the front
        return [self.tanh(output) for output in outputs]
        
class ModifiedUnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc,
                 submodule=None, norm_layer=nn.BatchNorm2d, use_dropout=False, styled=False, outermost=False, downsample_mode='strided', upsample_mode='transConv', upsample_method='nearest'):
        super(ModifiedUnetBlock, self).__init__()
        self.styled = styled
        if self.styled:
            use_bias = True
            self.adain = norm_layer(inner_nc if submodule is None else inner_nc * 2)
            self.up_activation = nn.ReLU(True)
            norm_layer = nn.BatchNorm2d
            if submodule is None:
                self.linear1 = nn.Sequential(nn.ReLU(True), nn.Linear(inner_nc, inner_nc))
                self.linear2 = nn.Sequential(nn.ReLU(True), nn.Linear(inner_nc, inner_nc))
        elif type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        downconv = getDownsample(outer_nc, inner_nc, 4, 2, 1, use_bias, downsample_mode=downsample_mode)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = [norm_layer(inner_nc)]
        uprelu = [nn.ReLU(True)] if not self.styled else []
        upnorm = [norm_layer(outer_nc)] if not self.styled else []

        if submodule is None:
            upconv = getUpsample(inner_nc, outer_nc, 4, 2, 1, use_bias, upsample_mode, upsample_method=upsample_method)
            down = [downrelu] + downconv
            up = uprelu + upconv + upnorm
        else:
            upconv = getUpsample(inner_nc * 2, outer_nc, 4, 2, 1, use_bias, upsample_mode, upsample_method=upsample_method)
            down = [downrelu] + downconv + downnorm
            up = uprelu + upconv + upnorm

            if use_dropout:
                up = up + [nn.Dropout(0.5)]

        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.submodule = submodule
        inconv = [nn.LeakyReLU(0.2, True), nn.Conv2d(input_nc, outer_nc, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(outer_nc, outer_nc, kernel_size=3, stride=1, padding=1), norm_layer(outer_nc)]
        self.inconv = nn.Sequential(*inconv)
        self.inconv_scalar = torch.nn.Parameter(torch.cuda.FloatTensor(1))
        self.inconv_scalar.requires_grad = True
        decimation = [nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),
            nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(input_nc), nn.LeakyReLU(0.2, True)]
        self.decimation = nn.Sequential(*decimation) if self.submodule is not None else None

    def forward(self, x, input):
        if self.styled:
            return self.styled_forward(x, input)
        transformed_input = self.inconv(input)
        residual_x = x + self.inconv_scalar * transformed_input
        submodule_x = self.down(residual_x)
        
        if self.submodule is None:
            submodule_outputs = [torch.cat([residual_x, self.up(submodule_x)], 1)]
        else:
            decimated_input = self.decimation(input)
            submodule_outputs, _ = self.submodule(submodule_x, decimated_input)
            feature_output = torch.cat([residual_x, self.up(submodule_outputs[-1])], 1)
            submodule_outputs.append(feature_output)
        return submodule_outputs, None

    def styled_forward(self, x, input):
        transformed_input = self.inconv(input)
        residual_x = x + self.inconv_scalar * transformed_input
        submodule_x = self.down(residual_x)
        if self.submodule is None:
            style = self.linear2(self.linear1(submodule_x.mean(-1).mean(-1)))
            intermediate = self.adain(self.up_activation(submodule_x), style)
            intermediate = self.up(intermediate)
            submodule_outputs = [torch.cat([residual_x, intermediate], 1)]
        else:
            decimated_input = self.decimation(input)
            submodule_outputs, style = self.submodule(submodule_x, decimated_input)
            intermediate = submodule_outputs[-1]
            intermediate = self.adain(self.up_activation(intermediate), style)
            feature_output = torch.cat([residual_x, self.up(intermediate)], 1)
            submodule_outputs.append(feature_output)

        return submodule_outputs, style

class MultiNLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, downsample_mode='strided', upsample_method='nearest'):
        super(MultiNLayerDiscriminator, self).__init__()
        self.n_layers = n_layers

        kw = 4
        padw = 1

        num_filters = [ndf * (2 ** min(i, 3)) for i in range(n_layers + 1)]
        sequence = []
        for n in range(n_layers):
            sequence.append(getDownsample(num_filters[n], num_filters[n + 1], kw, 2, padw, True, downsample_mode=downsample_mode) +
                [norm_layer(num_filters[n + 1]), nn.LeakyReLU(0.2, True)])

        output_map = []
        output_activation = [nn.Sigmoid()] if use_sigmoid else []
        for n in range(n_layers):
            nf = num_filters[n]
            output_map.append([
                nn.Conv2d(nf, nf, kernel_size=kw, stride=1, padding=padw, bias=False),
                norm_layer(nf), nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)] + output_activation)

        self.input_map = nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(ndf), nn.LeakyReLU(0.2, True))
        new_input_maps = [nn.Sequential(nn.Conv2d(input_nc, num_filters[i], kernel_size=1, stride=1, padding=0),
            norm_layer(num_filters[i]), nn.LeakyReLU(0.2, True)) for i in range(n_layers)]
        self.input_map_scalars = [torch.nn.Parameter(torch.cuda.FloatTensor(1)) for i in range(n_layers)]
        for i in range(n_layers):
            self.input_map_scalars[i].requires_grad = True
        self.decimation = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        
        for i in range(n_layers):
            setattr(self, 'sequence'+str(i), nn.Sequential(*sequence[i]))
            setattr(self, 'output_map'+str(i), nn.Sequential(*output_map[i]))
            setattr(self, 'new_input_maps'+str(i), new_input_maps[i])

    def forward(self, input, create_series=False):
        if create_series:
            return self.create_series(input)
        input1 = self.input_map(input[0])
        intermediate_features = input1
        outputs = []
        for i in range(self.n_layers):
            total = self.input_map_scalars[i] * getattr(self, 'new_input_maps'+str(i))(input[i]) + intermediate_features
            intermediate_features = getattr(self, 'sequence'+str(i))(total)
            outputs.append(getattr(self, 'output_map'+str(i))(total))

        return outputs

    def create_series(self, input):
        output = [input]
        for i in range(self.n_layers - 1):
            output.append(self.decimation(output[-1]))
        return output

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
