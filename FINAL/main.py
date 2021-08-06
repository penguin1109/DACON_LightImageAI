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
        #x = self.deconv(x)
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x