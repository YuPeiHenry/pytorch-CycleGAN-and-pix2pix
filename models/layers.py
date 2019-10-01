import torch
import torch.nn as nn

def getDownsample(in_c, out_c, k_size, stride, padding, use_bias, downsample_mode):
    if downsample_mode == 'downsample':
        return [nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=False),
                nn.Conv2d(in_c, out_c,
                    kernel_size=3, stride=1,
                    padding=1, bias=use_bias)]
    elif downsample_mode == 'subpixel':
        filter = []
        for i in range(stride ** 2):
            filter += [[
                [[1 if j == i else 0 for j in range(stride ** 2)]]
            ] for _ in range(in_c)]
        filter = torch.FloatTensor(filter).view(in_c * (stride ** 2), 1, stride, stride)
        filter.requires_grad = True
        strided_pooling = nn.Conv2d(in_c, in_c * (stride ** 2),
                    kernel_size=stride, stride=stride,
                    padding=0, groups=in_c, bias=False)
        with torch.no_grad():
            strided_pooling.weight = nn.Parameter(filter)
        return [strided_pooling,
                    nn.Conv2d(in_c * (stride ** 2), out_c,
                        kernel_size=3, stride=1,
                        padding=1, bias=use_bias)
                ]
    elif downsample_mode == 'max_pool':
        return [nn.Conv2d(in_c, out_c,
                    kernel_size=1, stride=1,
                    padding=0, bias=use_bias),
                    nn.MaxPool2d(stride)]
    else:
        return [nn.Conv2d(in_c, out_c,
                    kernel_size=k_size, stride=stride,
                    padding=padding, bias=use_bias)]

def getUpsample(in_c, out_c, k_size, stride, padding, use_bias, upsample_mode, upsample_method='nearest', output_padding=0):
    if upsample_mode == 'upsample':
        return [nn.Upsample(scale_factor=stride, mode=upsample_method),
                nn.Conv2d(in_c, out_c,
                    kernel_size=3, stride=1,
                    padding=1, bias=use_bias)]
    elif upsample_mode == 'subpixel':
        return [nn.Conv2d(in_c, out_c * (stride ** 2),
                    kernel_size=3, stride=1,
                    padding=1, bias=use_bias),
                    nn.PixelShuffle(stride)
                ]
    else:
        return [nn.ConvTranspose2d(in_c, out_c,
                    kernel_size=k_size, stride=stride,
                    padding=padding, output_padding=output_padding, bias=use_bias)]

#Taken from https://github.com/bfortuner/pytorch_tiramisu

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('leaky_relu', nn.LeakyReLU(0.2, True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels, downsample_mode='max_pool'):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('leaky_relu', nn.LeakyReLU(0.2, True))
        downsample = getDownsample(in_channels, in_channels, 4, 2, 1, True, downsample_mode=downsample_mode)
        downsample = nn.Sequential(*downsample)
        self.add_module('downsample', downsample)
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_mode='transConv'):
        super().__init__()
        if upsample_mode == 'transConv':
            convTrans = getUpsample(in_channels, out_channels,
                3, 1, 0, True, upsample_mode=upsample_mode)
        else:
            convTrans = getUpsample(in_channels, out_channels,
                3, 1, 1, True, upsample_mode=upsample_mode)
        self.convTrans = nn.Sequential(*convTrans)
        
    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]