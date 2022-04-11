from torch import nn

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

        # FCN Segmentation
        self.do = nn.Dropout()
        self.bnfcn = nn.BatchNorm2d(13) # Number of segmentation classes
        self.convfcn = nn.Conv2d(512, 13, kernel_size=1) 
        self.dcn32 = nn.ConvTranspose2d(13, 13, kernel_size=64, stride=32, dilation=1, padding=16)
        self.dcn32.weight.data = bilinear_init(self.cls_num, self.cls_num, 64)

        # UNet Segmentation
        self.umaxpool = nn.MaxPool2d(2)
        self.uconv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.ubn = nn.BatchNorm2d(mid_channels)

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

        if fcn32:
            x = self.bnfcn(self.do(self.relu(self.convfcn(x))))
            x = self.bnfcn(self.do(self.relu(self.dcn32(x))))
        else if unet:
            x = self.uconv(self.uconv(x))
            x = self.maxpool(2)

        return x

if __name__ == '__main__':
    import torch
    net = ResNet18Plain()
    y = net.forward(torch.randn(4, 64, 224, 224))
    print(y.size())
