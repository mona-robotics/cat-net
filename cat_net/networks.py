import torch
from torch import nn
from torch.autograd import Variable
from . import pytorch_ssim


class UNet(nn.Module):
    """Create a U-Net with skip connections."""

    def __init__(self, source_channels, output_channels, down_levels,
                 num_init_features=64, max_features=512, drop_rate=0,
                 innermost_kernel_size=None,
                 use_cuda=False):
        super().__init__()

        self.use_cuda = use_cuda

        # Initial convolution
        self.model = nn.Sequential()
        self.model.add_module('conv0',
                              nn.Conv2d(source_channels, num_init_features,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=True))
        down_levels = down_levels - 1  # We just downsampled one level
        total_features = num_init_features

        # Build the inner blocks recursively
        if down_levels > 0:
            submodule = SkipConnectionBlock(
                num_input_features=num_init_features,
                down_levels=down_levels,
                max_features=max_features,
                drop_rate=drop_rate,
                innermost_kernel_size=innermost_kernel_size)
            self.model.add_module('submodule', submodule)
            total_features += submodule.num_outer_features

        # Final convolution
        self.model.add_module('norm0',
                              nn.InstanceNorm2d(total_features, affine=True))
        self.model.add_module('relu0', nn.ReLU(inplace=True))
        self.model.add_module('conv1',
                              nn.ConvTranspose2d(total_features,
                                                 output_channels,
                                                 kernel_size=4, stride=2, padding=1, output_padding=0,
                                                 bias=True))
        self.model.add_module('tanh', nn.Tanh())

        if self.use_cuda:
            self.model = nn.DataParallel(self.model)

    def forward(self, x):
        return self.model(x)


class SkipConnectionBlock(nn.Sequential):
    def __init__(self, num_input_features, down_levels,
                 max_features, drop_rate, innermost_kernel_size):
        super().__init__()

        self.num_outer_features = num_input_features
        self.num_inner_features = min(num_input_features * 2, max_features)

        if down_levels == 1 and innermost_kernel_size is not None:
            # This is the innermost block
            kernel_size = innermost_kernel_size
        else:
            kernel_size = 4

        # Downsampling
        self.add_module('norm0',
                        nn.InstanceNorm2d(self.num_outer_features, affine=True))
        self.add_module('relu0', nn.LeakyReLU(0.2, inplace=True))
        self.add_module('conv0',
                        nn.Conv2d(self.num_outer_features,
                                  self.num_inner_features,
                                  kernel_size=kernel_size, stride=2, padding=1,
                                  bias=True))
        if drop_rate > 0 and self.num_inner_features == self.num_outer_features:
            self.add_module('dropout0', nn.Dropout2d(drop_rate))

        down_levels = down_levels - 1  # We just downsampled one level

        # Submodule
        if down_levels > 0:
            submodule = SkipConnectionBlock(
                num_input_features=self.num_inner_features,
                down_levels=down_levels,
                max_features=max_features,
                drop_rate=drop_rate,
                innermost_kernel_size=innermost_kernel_size)

            self.add_module('submodule', submodule)
            total_features = self.num_inner_features + \
                submodule.num_outer_features
        else:
            total_features = self.num_inner_features

        # Upsampling
        self.add_module('norm1',
                        nn.InstanceNorm2d(total_features, affine=True))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.ConvTranspose2d(total_features,
                                           self.num_outer_features,
                                           kernel_size=kernel_size, stride=2, padding=1, output_padding=0,
                                           bias=True))
        if drop_rate > 0 and self.num_inner_features == self.num_outer_features:
            self.add_module('dropout1', nn.Dropout2d(drop_rate))

    def forward(self, x):
        new_features = super().forward(x)
        return torch.cat([x, new_features], dim=1)

def set_sobel_x_weights(conv):
    conv.weight.data[:, :, 0, 0] = 0.5
    conv.weight.data[:, :, 0, 1] = 0
    conv.weight.data[:, :, 0, 2] = -0.5

    conv.weight.data[:, :, 1, 0] = 1
    conv.weight.data[:, :, 1, 1] = 0
    conv.weight.data[:, :, 1, 2] = -1

    conv.weight.data[:, :, 2, 0] = 0.5
    conv.weight.data[:, :, 2, 1] = 0
    conv.weight.data[:, :, 2, 2] = -0.5


def set_sobel_y_weights(conv):
    conv.weight.data[:, :, 0, 0] = 0.5
    conv.weight.data[:, :, 0, 1] = 1
    conv.weight.data[:, :, 0, 2] = 0.5

    conv.weight.data[:, :, 1, 0] = 0
    conv.weight.data[:, :, 1, 1] = 0
    conv.weight.data[:, :, 1, 2] = 0

    conv.weight.data[:, :, 2, 0] = -0.5
    conv.weight.data[:, :, 2, 1] = -1
    conv.weight.data[:, :, 2, 2] = -0.5


class GradientLoss(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()

        self.sobel_x = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1, bias=False)
        set_sobel_x_weights(self.sobel_x)

        self.sobel_y = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1, bias=False)
        set_sobel_y_weights(self.sobel_y)

        if use_cuda:
            self.sobel_x = self.sobel_x.cuda()
            self.sobel_y = self.sobel_y.cuda()

    def __call__(self, input, target):
        grad_x = self.sobel_x.forward(input)
        grad_y = self.sobel_y.forward(input)

        grad = (grad_x**2 + grad_y**2).sum()
        return -grad  # minimize the negative of the gradient magnitude

class GradientLoss2(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()

        self.sobel_x = torch.nn.Conv2d(
            3, 3, 3, stride=1, padding=1, bias=False)
        set_sobel_x_weights(self.sobel_x)

        self.sobel_y = torch.nn.Conv2d(
            3, 3, 3, stride=1, padding=1, bias=False)
        set_sobel_y_weights(self.sobel_y)

        if use_cuda:
            self.sobel_x = self.sobel_x.cuda()
            self.sobel_y = self.sobel_y.cuda()

    def __call__(self, input, target):
        grad_x_input = self.sobel_x.forward(input)
        grad_y_input = self.sobel_y.forward(input)

        grad_x_target = self.sobel_x.forward(target)
        grad_y_target = self.sobel_y.forward(target)

        grad_input = (grad_x_input**2 + grad_y_input**2).sum()
        grad_target = (grad_x_target**2 + grad_y_target**2).sum()
        return grad_target - grad_input  # minimize the negative of the gradient magnitude

class GradientLoss3(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()

        self.sobel_x = torch.nn.Conv2d(
            3, 3, 3, stride=1, padding=1, bias=False)
        set_sobel_x_weights(self.sobel_x)

        self.sobel_y = torch.nn.Conv2d(
            3, 3, 3, stride=1, padding=1, bias=False)
        set_sobel_y_weights(self.sobel_y)

        if use_cuda:
            self.sobel_x = self.sobel_x.cuda()
            self.sobel_y = self.sobel_y.cuda()

    def __call__(self, input, target):
        grad_x_input = self.sobel_x.forward(input)
        grad_y_input = self.sobel_y.forward(input)

        grad_x_target = self.sobel_x.forward(target)
        grad_y_target = self.sobel_y.forward(target)

        return (((grad_x_input - grad_x_target).abs() + (grad_y_input - grad_y_target).abs()).sum() /  input.data.nelement()) 

class L1GradLoss(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()
        self.loss_l1 = nn.L1Loss()
        self.loss_grad = GradientLoss3(use_cuda)

    def __call__(self, input, target):

        l1_loss = self.loss_l1(input, target)
        grad_loss = self.loss_grad(input, target)
        return l1_loss + grad_loss

class L1SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_l1 = nn.L1Loss()
        self.loss_ssim = pytorch_ssim.SSIM(window_size = 11)

    def __call__(self, input, target):

        l1_loss = self.loss_l1(input, target)
        ssim_loss = -self.loss_ssim(input, target)
        return l1_loss + ssim_loss

class RMSLogLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input, target):

        y = Variable(torch.Tensor([2]).cuda(async=True).float()) # numpy is double by default
        tmp_input = input + y.expand(input.size())
        tmp_target = target + y.expand(target.size())
        return (((tmp_input.log() - tmp_target.log()) ** 2).sum() / input.data.nelement()).sqrt()
