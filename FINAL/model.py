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