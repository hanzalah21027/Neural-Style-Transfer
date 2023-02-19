import torch
from torch import nn

# Generator

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels), 
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # c7s1-64 Block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # d128 Block
        model += [
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
            ]
        
        # d256 Block
        model += [
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
            ]
        
        # R256 Blocks
        for i in range(9):
            model += [ResidualBlock(in_channels = 256)]
            
        # u128 Block   
        model += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            ]

        # u64 Blocks
        model += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=True)
            ]
            
        # c7s1-3 Block
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 7),
                  nn.Tanh()
                 ]

        self.model = nn.Sequential(*model) 
        
    def forward(self, x):
        return self.model(x)

# Discriminator

class DiscriminatorBlock(nn.Module):
  def __init__(self, in_channels, out_channels, normalise = True):
    super(DiscriminatorBlock, self).__init__()
    Layers = [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 2, padding = 1)]
    if normalise:
      Layers += [nn.InstanceNorm2d(out_channels)] 
    Layers += [nn.LeakyReLU(negative_slope = 0.2, inplace=True)]
    self.model = nn.Sequential(*Layers)

  def forward(self, x):
    return self.model(x)
    

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    Layer = [DiscriminatorBlock(in_channels = 3, out_channels = 64, normalise = False)]
    Layer += [DiscriminatorBlock(in_channels = 64, out_channels = 128)]
    Layer += [DiscriminatorBlock(in_channels = 128, out_channels = 256)]
    Layer += [DiscriminatorBlock(in_channels = 256, out_channels = 512)]
    Layer += [nn.ZeroPad2d((1, 0, 1, 0)),
              nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, padding = 1)]
    self.model = nn.Sequential(*Layer)
        
  def forward(self, x):
    return self.model(x)


# Discriminator 2

class DiscriminatorV2(nn.Module):
    def __init__(self):
        super(DiscriminatorV2, self).__init__()
        
        def block(in_channels, out_channels, normalize=False):
            L = [nn.Conv2d(in_channels, out_channels, 4, stride = 2, padding=1)]
            if normalize:
                L += [nn.InstanceNorm2d(out_channels)]
            L += [nn.LeakyReLU(0.2, inplace=True)]
            return L
        
        self.model = nn.Sequential(
            *block(in_channels = 3, out_channels = 64),
            *block(in_channels = 64, out_channels = 128, normalize=True),
            *block(in_channels = 128,out_channels = 256, normalize=True),
            *block(in_channels = 256,out_channels = 512, normalize=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, padding=1)
        )
        
    def forward(self, x):
        return self.model(x)