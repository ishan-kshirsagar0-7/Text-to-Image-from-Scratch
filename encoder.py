# THIS IS THE ENCODER COMPONENT OF THE VARIATIONAL AUTOENCODER BLOCK.
# Encoding in VAE involves basically reducing the dimensionality of the input image (or noise) but increasing the features.
# The output of a VAE encoder is a latent space where each image is represented as a vector, distinguishable from other vectors.

import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch_size, channel, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),
            # (batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (batch_size, 128, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256), 
            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256), 
            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            # (batch_size, 256, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_AttentionBlock(512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.GroupNorm(32, 512),
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.SiLU(),
            # Because the padding=1, it means the width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8) 
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (batch_size, 8, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, channel, height, width)
        # noise : (batch_size, out_channel, height/8, width/8)
        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric
                # Pad: (padding_left, padding_right, padding_top, padding_bottom)
                # Pad with zeros on the right and bottom.
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (batch_size, 8, height/8, width/8) -> two tensors of shape (batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # We now clamp the log variance to some given range, in case it's too small or too big.
        # The shape of the tensor still remains the same, i.e, (batch_size, 4, height/8, width/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # We remove the log by raising log_variance exponentially. The shape of the tensor remains the same.
        variance = log_variance.exp()

        # Now we compute the Standard Deviation from the Variance. Again, the shape of the tensor remains the same.
        stdev = variance.sqrt()
        
        # We now have the mean as well as the variance of the distribution. But how do we sample from it?
        # In order to do so, we first assume a sample Z, from a standard normal distribution -> N(0,1)
        # Given Z, and the mean and the variance of the desired distribution, we can derive a sample "X" from this with the formula :
        # X = mean + stdev * Z
        # This is known as reparameterization in order to derive a sample from the desired distribution.
        x = mean + stdev * noise
        
        # Finally, we scale the output by a constant (as defined in the original Stable Diffusion paper)
        x *= 0.18215
        
        return x

