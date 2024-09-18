import torch
import torch.nn as nn
from torch.optim import lr_scheduler


def get_norm_layer(opt):
    if opt["norm"] == "instance":
        norm_layer = nn.InstanceNorm2d
    elif opt["norm"].lower() == "none":
        norm_layer = nn.Identity
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % opt["norm"])
    return norm_layer


def init_net(net, opt):
    device = torch.device(opt["device"])
    if device == torch.device("cuda"):
        net = nn.DataParallel(net)
    net.to(device)
    return net


def define_resnet(opt, input):
    """input (str) -- input domain of the generator (A or B)"""
    input_nc = opt["In_nc"]
    output_nc = opt["Out_nc"]
    net = ResnetGenerator(opt, input_nc, output_nc)
    return init_net(net, opt)


class ResnetGenerator(nn.Module):
    """Create a generator

    Parameters:
        opt["input_nc"] (int) -- the number of channels in input images
        opt["output_nc"] (int) -- the number of channels in output images
        opt.ngf (int) -- the number of filters in the last conv layer
        #netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        opt.norm (str) -- the name of normalization layers used in the network: instance
        #use_dropout (bool) -- if use dropout layers.
        #init_type (str)    -- the name of our initialization method.
        #init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        #gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """

    def __init__(self, opt, input_nc, output_nc):
        super(ResnetGenerator, self).__init__()
        norm_layer = get_norm_layer(opt)
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, opt["ngf"], kernel_size=7, padding=0, bias=True),
                 norm_layer(opt["ngf"]),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(opt["ngf"] * mult, opt["ngf"] * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                      norm_layer(opt["ngf"] * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(9):
            model += [ResnetBlock(opt["ngf"] * mult, norm_layer=norm_layer)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(opt["ngf"] * mult, int(opt["ngf"] * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                      norm_layer(int(opt["ngf"] * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(opt["ngf"], output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """ Define a Resnet block"""

    def __init__(self, dim, norm_layer):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer)

    def build_conv_block(self, dim, norm_layer):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       norm_layer(dim),
                       nn.ReLU(True),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)  # unpacks the list into positional arguments

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)
        return out


def define_patchGAN(opt):
    input_nc = opt["In_nc"]
    output_nc = opt["Out_nc"]
    net = PatchGANDiscriminator(opt, input_nc + output_nc)
    return init_net(net, opt)


class PatchGANDiscriminator(nn.Module):
    """Create a PatchGAN discriminator"""

    def __init__(self, opt, input_nc, ndf=64, n_layers=3):
        """ Construct the PatchGAN discriminator

        Parameters:
            input_nc (int)     -- the number of channels in input images
            ndf (int)          -- the number of filters in the first conv layer
            norm (str)         -- the type of normalization layers used in the network.

        Returns a discriminator
        """
        super(PatchGANDiscriminator, self).__init__()
        norm_layer = get_norm_layer(opt)
        model = [nn.Conv2d(input_nc, opt["ngf"], kernel_size=4, stride=2, padding=1, bias=True),
                 nn.LeakyReLU(0.2, True)]

        for i in range(n_layers-1):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(opt["ngf"] * mult, opt["ngf"] * mult * 2, kernel_size=4, stride=2, padding=1, bias=True),
                      norm_layer(opt["ngf"] * mult * 2),
                      nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(opt["ngf"] * (2**(n_layers-1)), opt["ngf"] * (2**n_layers), kernel_size=4, stride=1, padding=1, bias=True),
                  norm_layer(opt["ngf"] * (2**n_layers)),
                  nn.LeakyReLU(0.2, True),
                  nn.Conv2d(opt["ngf"] * (2**n_layers), 1, kernel_size=4, stride=1, padding=1, bias=True)]  # 1 channel prediction output

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


def get_scheduler(optimizer, opt):
    def lambda_decay(epoch):
        decay = 1.0 - max(0, opt["epoch_count"] - opt["learning_rate_fix"]) / float(opt["learning_rate_decay"] + 1)  # epoch is taken from opt to include training resume
        return decay

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_decay)
    return scheduler


def define_PixelDiscriminator(opt, input):
    input_nc = opt["In_nc"]
    net = PixelDiscriminator(opt, input_nc)
    return init_net(net, opt)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, opt, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


def define_discriminator(opt):
    try:
        if opt["discriminator"] == "PatchGANDiscriminator":
            return define_patchGAN
        elif opt["discriminator"] == "PixelDiscriminator":
            return define_PixelDiscriminator
        else:
            raise NotImplementedError("Discriminator not implemented in networks.py.")
    except NotImplementedError:
        raise
