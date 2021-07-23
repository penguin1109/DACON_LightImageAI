import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            # 각 채널별로 average pooling을 진행한다 (채널 수가 c개이면 c개 모두에 대해서 진행)
            nn.AdaptiveAvgPool2d(1),
            # downsampling                                             
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            # upsampling
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
  # RIR(Residual in Residual) 구조를 적용
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
      # x는 RCAB의 input, output과 element-wise sum을 적용
      return x + self.module(x)


class RG(nn.Module):
  # RG(Residual Group)
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        # num_recab = 12(하나의 RG안에 RCAB이 12개 존재)
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)

class RCAN(nn.Module):
    def __init__(self, config):
        super(RCAN, self).__init__()
        scale = config['scale']
        num_features = config['num_features']
        # number of residual groups
        num_rg = config['num_rg']
        num_rcab = config['num_rcab']
        reduction = config['reduction']
        
        # 먼저 low resolution의 이미지를 4배로 늘려주어 적용시킨다.
        self.deconv = nn.ConvTranspose2d(3, 3, kernel_size=9, stride=4, padding=4, output_padding=3)
        # 본격적인 특성 추출에 앞서서 표면적인 특성을 추출하는 하나의 convolution layer
        self.sf = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        # RIR, 즉 residual in residual모듈을 적용함
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])

        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)
        # 0.0 ~ 1.0 사이의 픽셀값을 갖는 이미지의 출력을 위해 sigmoid적용
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # self.upscale은 원래 pixel shuffle을 통해서 channel의 수를 r**2배
        # 가로 세로 길이는 r배로 늘림
        # 그러나 그렇게 크기를 키우는 과정은 필요가 없고 
        # 오히려 이미지의 channel에 집중하여 하는 것이 낫다.
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x