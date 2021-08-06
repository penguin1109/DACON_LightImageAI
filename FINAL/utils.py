from collections import namedtuple
import torch
import torchvision.models as models

class VGG16(torch.nn.Module):
    def __init__(self, requires_grad = False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained = True)
        self.block1 = torch.nn.Sequential()
        self.block2 = torch.nn.Sequential()
        self.block3 = torch.nn.Sequential()
        self.block4 = torch.nn.Sequential()

        for i in range(4):
            self.block1.add_module(str(i), vgg_pretrained_features[i])
        for i in range(4, 9):
            self.block2.add_module(str(i), vgg_pretrained_features[i])
        for i in range(9, 16):
            self.block3.add_module(str(i), vgg_pretrained_features[i])
        for i in range(16, 23):
            self.block4.add_module(str(i), vgg_pretrained_features[i])

        if requires_grad == False:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        h = self.block1(x)
        h_relu1_2 = h
        h = self.block2(x)
        h_relu2_2 = h
        h = self.block3(x)
        h_relu3_2 = h
        h = self.block4(x)
        h_relu4_2 = h

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_2', 'relu4_2'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_2, h_relu4_2)

        return out


def gram_matrix(x):
    (b, ch, h, w) = x.size()
    feat = x.view(b, ch, w*h)
    feat_t = feat.transpose(1, 2)
    gram = feat.bmm(feat_t) / (ch * w * h)

    return gram

def normalize(x):
    mean = x.new_tensor([0.485, 0.456, 0.406].view(-1, 1, 1))
    std = x.new_tensor([0.229, 0.224, 0.225].view(-1, 1, 1))
    return (x-mean)/std



