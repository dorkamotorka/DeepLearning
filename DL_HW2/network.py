import torch
from torch import nn
import numpy as np
import torchvision
import torch.nn.functional as F

def bilinear_init(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt

    return torch.from_numpy(weight)

# Initialize 18-layer plain(without residual) ResNet
class ResNet18Plain(nn.Module):
    def __init__(self, fcn32=False):
        super(ResNet18Plain, self).__init__()
        # (N - F + 2P) / stride + 1
        # N - size of the image (NxN)
        # F - size of the filter (FxF)
        # P - padding
        # stride ~ korak
        # Batch normalization after every convolution
        # ReLU only after every convolution, expect for the last one
        self.relu = nn.ReLU()

        # Initial layer
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3) 
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1) # Downsampling 112x112 -> 56x56

        # Block 1 - repeats twice
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        # Block 2 - repeats twice
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # Downsampling 56x56 -> 28x28
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        # Block 3 - repeats twice
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # Downsampling 28x28 -> 14x14
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        # Block 4 - repeats twice
        self.bn5 = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(256, 512, 3, stride=2, padding=1) # Downsampling 14x14 -> 7x7
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        # Output Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 400)


    def forward(self, x):
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Layer 2 - 5
        x = self.conv2_1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2(x)

        # Layer 6 - 9
        x = self.conv3_1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3(x)

        # Layer 10 - 13
        x = self.conv4_1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4(x)

        # Layer 14 - 17
        x = self.conv5_1(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5(x)

        # Layer 18
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

class ResNet18FCN(nn.Module):
    def __init__(self):
        super(ResNet18FCN, self).__init__()
        # (N - F + 2P) / stride + 1
        # N - size of the image (NxN)
        # F - size of the filter (FxF)
        # P - padding
        # stride ~ korak
        # Batch normalization after every convolution
        # ReLU only after every convolution, expect for the last one
        self.relu = nn.ReLU()

        # Initial layer
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3) 
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1) # Downsampling 112x112 -> 56x56

        # Block 1 - repeats twice
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        # Block 2 - repeats twice
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # Downsampling 56x56 -> 28x28
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        # Block 3 - repeats twice
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(128, 256, 3, stride=2, padding=1) # Downsampling 28x28 -> 14x14
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        # Block 4 - repeats twice
        self.bn5 = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(256, 512, 3, stride=2, padding=1) # Downsampling 14x14 -> 7x7
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        # Output Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 400)

        # FCN Segmentation
        self.do = nn.Dropout()
        self.bnfcn = nn.BatchNorm2d(13) # Number of segmentation classes
        self.convfcn_1 = nn.Conv2d(512, 512, kernel_size=1) 
        self.convfcn_2 = nn.Conv2d(512, 13, kernel_size=1) 
        self.dcn32 = nn.ConvTranspose2d(13, 13, kernel_size=64, stride=32, dilation=1, padding=16)
        self.dcn32.weight.data = bilinear_init(13, 13, 64)

    def forward(self, x):
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Layer 2 - 5
        x = self.conv2_1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2(x)

        # Layer 6 - 9
        x = self.conv3_1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3(x)

        # Layer 10 - 13
        x = self.conv4_1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4(x)

        # Layer 14 - 17
        x = self.conv5_1(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5(x)

        # Layer 18
        # In FCN discard output layer 
        #x = self.avgpool(x)
        #x = x.reshape(x.shape[0], -1)
        #x = self.fc(x)

        # FCN additional blocks
        x = self.convfcn_1(x)
        x = self.relu(x)
        x = self.do(x)
        #x = self.bnfcn(x)
        x = self.relu(x)
        x = self.do(x)
        #x = self.bnfcn(x)

        x = self.convfcn_2(x)
        x = self.dcn32(x)

        return x

class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=13, out_sz=(320,416)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        # Add padding to images to persist dimensions
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)

        # Retain dimension
        out = F.interpolate(out, self.out_sz)

        return out

class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)

        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)

        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)

        return enc_ftrs

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

if __name__ == '__main__':
    import torch
    net = ResNet18Plain()
    y = net.forward(torch.randn(4, 3, 224, 224))
    print(y.size())

    unet = UNet()
    x = torch.randn(4, 3, 572, 572)
    print(unet(x).shape)
