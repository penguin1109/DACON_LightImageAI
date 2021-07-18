import torch
import torch.nn as nn

class Convolution_Block(nn.Module):
    def __init__(self, in_dim, out_dim, activation = True):
        super(Convolution_Block, self).__init__()
        if activation:
            self.convolutionblock = nn.Sequential(
                nn.Conv2d(in_channels = in_dim, out_channels = out_dim, kernel_size = 3,
                strides = (1, 1), padding = 'same'),
                nn.LeakyReLU(negative_slope = 0.2)
        )
        else:
            self.convolutionblock = nn.Conv2d(
                in_channels = in_dim, out_channels = out_dim, kernel_size = 3,
                strides = (1, 1), padding = 'same'
            )
        self.BN = nn.BatchNorm2d()
    
    def forward(self, x):
        out = self.convolutionblock(x)
        out = self.BN(out)

        return out
    
class Residual_Block(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(Residual_Block, self).__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.residualblock = nn.Sequential(
            nn.Conv2d(in_channels = self.in_dim, out_channels = self.mid_dim, kernel_size = 3, padding = 1),
            nn.ReLU,
            nn.Conv2d(in_channels = self.mid_dim, out_channels = self.out_dim, kernel_size = 3, paddimg = 1)
        )
        # 굳이 이렇게 지정해 놓는 이유는 학습을 시킬 때에 해당 block의 ReLU activation function을
        # 계속 이용해야 하는 상황이기 때문이다.
        self.relu = nn.ReLU
    
    def forward(self, x):
        out = self.residualblock(x)
        out += x
        out = self.relu(out)
        return out