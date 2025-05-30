import torch.nn as nn
import torch

import torch.nn.utils.spectral_norm as spectral_norm

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle. Source: https://github.com/kuoweilai/pixelshuffle3d
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)
    

class Generator(nn.Module):
    def __init__(self, lr_res = 5, scale_factor = 2, n_residual_blocks = 3, n_features = 32, inner_kernel_size = 3, outer_kernel_size = 5):
        super(Generator, self).__init__()
        self.input_size = lr_res
        self.scale_factor = scale_factor
        self.n_residual_blocks = n_residual_blocks

        self.input_conv = nn.Sequential(
            nn.Conv3d(4, n_features, outer_kernel_size, stride=1, padding='same'),
            nn.LeakyReLU(0.2)
        )

        self.residual_blocks = nn.ModuleList()
        for _ in range(n_residual_blocks):
            self.residual_blocks.append(self.residual_block(n_features, n_features, inner_kernel_size))

        self.intermediate_conv = nn.Sequential(
            nn.Conv3d(n_features, n_features, outer_kernel_size, stride=1, padding='same'),
            nn.BatchNorm3d(n_features)
        )

        self.upsample = nn.Sequential(
            nn.Conv3d(n_features, n_features, inner_kernel_size, stride=1, padding='same'),
            PixelShuffle3d(scale_factor),
            nn.LeakyReLU(0.2)
        )

        self.output_conv = nn.Sequential(
            nn.Conv3d(n_features//(scale_factor**3), 4, outer_kernel_size, 1, padding='same'),
            nn.Tanh()
        )

    def residual_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding='same'),
            nn.BatchNorm3d(out_channels)
        )
    
    def forward(self, x):
        x = self.input_conv(x)
        residual = x
        for block in self.residual_blocks:
            x = block(x)
            x = x + residual
        x = self.intermediate_conv(x)
        x = x + residual
        x = self.upsample(x)
        x = self.output_conv(x)

        return x
    

    

class Discriminator(nn.Module):
    def __init__(self, kernel_size = 3, channel_nums = [64, 64, 128], stride_sizes = [1,2,1], dense_size = 512, conditional = False):
        super(Discriminator, self).__init__()

        self.conditional = conditional
        
        if len(channel_nums) != len(stride_sizes):
            raise ValueError("channel_nums and stride_sizes must have the same length")
        
        self.input_conv = nn.Sequential(
            nn.Conv3d(4, channel_nums[0], kernel_size, stride=stride_sizes[0], padding=(kernel_size//2)),
            nn.LeakyReLU(0.2)
        )

        self.conv_blocks = nn.Sequential(
            *[self.conv_block(channel_nums[i], channel_nums[i+1], kernel_size, stride_sizes[i+1]) for i in range(len(channel_nums)-1)]
        )

        #Flattening option 1:
        # self.flatten = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.AdaptiveAvgPool3d(1))
        
        # flat_size = channel_nums[-1]
        
        #Flattening option 2:
        self.flatten = nn.Sequential(
           nn.Dropout(0.5),
           nn.Flatten()
        )

        flat_size = channel_nums[-1]*(5**3)



        if conditional:
            self.linear1 = nn.Linear(flat_size, 1, bias=True)
            nn.init.xavier_uniform_(self.linear1.weight)
            self.linear2 = nn.Linear(1, flat_size, bias=False)
            nn.init.xavier_uniform_(self.linear2.weight)
            self.sigmoid = nn.Sigmoid()
        else:
            self.linear1 = nn.Linear(flat_size, dense_size)
            nn.init.xavier_uniform_(self.linear1.weight)
            self.linear2 = nn.Linear(dense_size, 1)
            nn.init.xavier_uniform_(self.linear2.weight)

    
            self.output_net = nn.Sequential(
                self.linear1,
                nn.LeakyReLU(0.2),
                self.linear2,
                nn.Sigmoid()
            )

    def conv_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding = kernel_size//2),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x, condition=None):
        x = self.input_conv(x)
        x = self.conv_blocks(x)
        x = self.flatten(x).view(x.size(0), -1)

        if self.conditional and condition is not None:
            inner_mult = torch.sum(x*self.linear2(condition.unsqueeze(1)+1), 1, keepdim=True)
            output = self.sigmoid(self.linear1(x) + inner_mult)

        else:
            output = self.output_net(x)

        return output


