import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, complexity=64):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.complexity = complexity

        self.inc = nn.Sequential(
            nn.Conv2d(input_channels, complexity, kernel_size=3, padding=(1, 1), bias=False),
            nn.BatchNorm2d(complexity),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(complexity, complexity*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(complexity*2, complexity*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(complexity*4, complexity*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(complexity*8, complexity*16))
        self.down5 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(complexity*16, complexity*32))
        self.down6 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(complexity*32, complexity*64))
        
        # Bridge
        self.bridge = nn.Sequential(
            ResidualBlock(complexity*64, complexity*64),
            ResidualBlock(complexity*64, complexity*64)
        )
        
        self.up1 = nn.ConvTranspose2d(complexity*64, complexity*32, kernel_size=2, stride=2)
        self.conv1 = ResidualBlock(complexity*64, complexity*32)
        self.up2 = nn.ConvTranspose2d(complexity*32, complexity*16, kernel_size=2, stride=2)
        self.conv2 = ResidualBlock(complexity*32, complexity*16)
        self.up3 = nn.ConvTranspose2d(complexity*16, complexity*8, kernel_size=2, stride=2)
        self.conv3 = ResidualBlock(complexity*16, complexity*8)
        self.up4 = nn.ConvTranspose2d(complexity*8, complexity*4, kernel_size=2, stride=2)
        self.conv4 = ResidualBlock(complexity*8, complexity*4)
        self.up5 = nn.ConvTranspose2d(complexity*4, complexity*2, kernel_size=2, stride=2)
        self.conv5 = ResidualBlock(complexity*4, complexity*2)
        self.up6 = nn.ConvTranspose2d(complexity*2, complexity, kernel_size=2, stride=2)
        self.conv6 = ResidualBlock(complexity*2, complexity)
        
        self.dec = nn.Conv2d(complexity, output_channels, kernel_size=3, padding=(1, 1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        
        # Bridge
        x7 = self.bridge(x7)
        
        x = self.up1(x7)
        x = torch.cat([x, x6], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x5], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x4], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv4(x)
        x = self.up5(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv5(x)
        x = self.up6(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv6(x)
        x = self.dec(x)
        
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 使用示例
def main():
    # 创建不同复杂度的模型实例
    model_low = UNet(output_channels=2, complexity=8)
    model_medium = UNet(output_channels=2, complexity=16)
    model_high = UNet(output_channels=2, complexity=32)
    
    # 生成一个随机的514x544的单通道图像作为输入
    input_image = torch.randn(1, 1, 512, 512)
    
    # 对每个模型进行前向传播并打印参数量
    for name, model in [("Low", model_low), ("Medium", model_medium), ("High", model_high)]:
        output = model(input_image)
        print(f"{name} complexity model:")
        print(f"  Input shape: {input_image.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Total parameters: {count_parameters(model):,}")
        print()

if __name__ == "__main__":
    main()