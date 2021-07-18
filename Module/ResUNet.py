from Module.Blocks import Residual_Block
import torch.nn as nn
import torchvision.models as models

resnet152 = models.resnet152(pretrained = True)
class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()
        resnet152 = models.resnet152(pretrained = True)

        modules = list(resnet152.children())[:-3]
        resnet152 = nn.Sequential(*modules)

        for param in resnet152.parameters():
            param.requires_grad = False
    
        self.backbone = resnet152
        # 우선 resnet152의 마지막 output linear layer은 sigmoid를 적용하면 classification을 위한
        # 확률 값을 반환하게 될텐데 그 전 단계인 1demension의 이미지의 feature map를 
        # 우리가 구상하게 될 downsample-upsample구조에 넣어서 
        # label image와의 차이를 보아야 한다

        # 따라서 resnet152는 학습된 parameter을 이용할 것이기 때문에 gradient를 적용할 필요가 없다
        # 즉, weights를 학습 시킬 필요가 없다는 뜻
        def block(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1,bias = True):
            layers = []
            layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels,    kernel_size = kernel_size,
                           stride = stride, padding = padding, bias = bias)]
            layers += [nn.BatchNorm2d(num_features = out_channels)]
            layers += [nn.ReLU(inplace = True)]

            return nn.Sequential(*layers)

    # Contracting_path
    # (2048, 1, 1)의 shape를 넣어줌
        self.enc1_1 = block(in_channels = 1, out_channels = 64)
        self.enc1_2 = block(in_channels = 64, out_channels = 64)

        self.pool1 = nn.MaxPool2d(kernel_size = 2)

        self.enc2_1 = block(in_channels = 64, out_channels = 128)
        self.enc2_2 = block(in_channels = 128, out_channels = 128)

        self.pool2 = nn.MaxPool2d(kernel_size = 2)

        self.enc3_1 = block(in_channels = 128, out_channels = 256)
        self.enc3_2 = block(in_channels = 256, out_channels = 256)

    
    # Expansive_path
        self.dec3_1 = block(in_channels = 256, out_channels = 128)

        self.unpool2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 2,
                                      stride = 2, padding = 0, bias = True)
    
        self.dec2_2 = block(in_channels = 256, out_channels = 128)
        self.dec2_1 = block(in_channels = 128, out_channels = 64)

        self.unpool1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, 
                                      stride = 2, kernel_size = 2, padding = 0, bias = True)
    
        self.dec1_2 = block(in_channels = 128, out_channels = 64)
        self.dec1_1 = block(in_channels = 64, out_channels = 64)
    
        self.out = nn.Conv2d(in_channels = 64, out_channels = 3, stride = 1, kernel_size = 1, padding = 0, bias = True)
        self.outpool = nn.MaxPool2d(kernel_size = 2)
    
    def forward(self, x):
        x = self.backbone(x).reshape(-1, 1, 512, 512)

        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)

        dec3_1 = self.dec3_1(enc3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim = 1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim = 1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        output = self.out(dec1_1)
        output = self.outpool(output)

        return output


class ResUNet(nn.Module):
    def __init__(self, weight = None):
        super(ResUNet, self).__init__()
        resnet152 = models.resnet152(pretrained = True)
        resnet152 = nn.Sequential(*list(resnet152.children[:-1]))

        for param in resnet152.parameters():
            param.requires_grad = False
        
        # 우선 resnet152의 마지막 output linear layer은 sigmoid를 적용하면 classification을 위한
        # 확률 값을 반환하게 될텐데 그 전 단계인 1demension의 이미지의 feature map를 
        # 우리가 구상하게 될 downsample-upsample구조에 넣어서 
        # label image와의 차이를 보아야 한다

        # 따라서 resnet152는 학습된 parameter을 이용할 것이기 때문에 gradient를 적용할 필요가 없다
        # 즉, weights를 학습 시킬 필요가 없다는 뜻

        self.backbone = resnet152
        if weight:
            self.weigth = weight
    
    def forward(self, x):
        input_layer = self.backbone(x)

        conv4 = nn.LeakyReLU(negative_slope = 0.1)(input_layer)
        conv4 = nn.MaxPool2d(kernel_size = (2, 2))(conv4)
        pool4 = nn.Dropout(p = 0.1)(conv4)

        convm = nn.Conv2d(in_channels = start_neurons, out_channels = start_neurons * 16)(convm)
        convm = Residual_Block(convm, start_neurons * 16)



    

