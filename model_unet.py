import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, complexity=8, use_transformer=True):
        super(UNet, self).__init__()
        
        self.complexity = complexity
        self.use_transformer = use_transformer
        
        # Encoder
        self.enc_conv1 = self.conv_block(input_channels, complexity)
        self.enc_conv2 = self.conv_block(complexity, complexity*2)
        self.enc_conv3 = self.conv_block(complexity*2, complexity*4)
        self.enc_conv4 = self.conv_block(complexity*4, complexity*8)
        self.enc_conv5 = self.conv_block(complexity*8, complexity*16)
        self.enc_conv6 = self.conv_block(complexity*16, complexity*32)
        self.enc_conv7 = self.conv_block(complexity*32, complexity*64)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Transformer
        if use_transformer:
            self.pos_encoder = PositionalEncoding(complexity*64)
            self.transformer_block = TransformerBlock(complexity*64, nhead=2)
        
        # Decoder
        self.upconv7 = nn.ConvTranspose2d(complexity*64, complexity*32, 2, stride=2)
        self.dec_conv7 = self.conv_block(complexity*64, complexity*32)
        self.upconv6 = nn.ConvTranspose2d(complexity*32, complexity*16, 2, stride=2)
        self.dec_conv6 = self.conv_block(complexity*32, complexity*16)
        self.upconv5 = nn.ConvTranspose2d(complexity*16, complexity*8, 2, stride=2)
        self.dec_conv5 = self.conv_block(complexity*16, complexity*8)
        self.upconv4 = nn.ConvTranspose2d(complexity*8, complexity*4, 2, stride=2)
        self.dec_conv4 = self.conv_block(complexity*8, complexity*4)
        self.upconv3 = nn.ConvTranspose2d(complexity*4, complexity*2, 2, stride=2)
        self.dec_conv3 = self.conv_block(complexity*4, complexity*2)
        self.upconv2 = nn.ConvTranspose2d(complexity*2, complexity, 2, stride=2)
        self.dec_conv2 = self.conv_block(complexity*2, complexity)
        # self.upconv1 = nn.ConvTranspose2d(complexity, complexity//2, 2, stride=2)
        self.dec_conv1 = self.conv_block(complexity + input_channels, complexity//2)
        
        self.final_conv = nn.Conv2d(complexity//2, output_channels, 1)
    
    def conv_block(self, input_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc_conv1(x)
        e2 = self.enc_conv2(self.pool(e1))
        e3 = self.enc_conv3(self.pool(e2))
        e4 = self.enc_conv4(self.pool(e3))
        e5 = self.enc_conv5(self.pool(e4))
        e6 = self.enc_conv6(self.pool(e5))
        e7 = self.enc_conv7(self.pool(e6))
        
        # Transformer
        if self.use_transformer:
            b, c, h, w = e7.size()
            e7 = e7.view(b, c, -1).permute(2, 0, 1)  # (seq, batch, feature)
            e7 = self.pos_encoder(e7)
            e7 = self.transformer_block(e7)
            e7 = e7.permute(1, 2, 0).view(b, c, h, w)
        
        # Decoder
        d7 = self.dec_conv7(torch.cat([self.upconv7(e7), e6], 1))
        d6 = self.dec_conv6(torch.cat([self.upconv6(d7), e5], 1))
        d5 = self.dec_conv5(torch.cat([self.upconv5(d6), e4], 1))
        d4 = self.dec_conv4(torch.cat([self.upconv4(d5), e3], 1))
        d3 = self.dec_conv3(torch.cat([self.upconv3(d4), e2], 1))
        d2 = self.dec_conv2(torch.cat([self.upconv2(d3), e1], 1))
        d1 = self.dec_conv1(torch.cat([d2, x], 1))
        return self.final_conv(d1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Discriminator(nn.Module):
    def __init__(self, input_channels=2, complexity=64):
        super(Discriminator, self).__init__()
        self.complexity = complexity
        
        # 定义卷积层
        self.conv1 = self.conv_block(input_channels, complexity)
        self.conv2 = self.conv_block(complexity, complexity*2)
        self.conv3 = self.conv_block(complexity*2, complexity*4)
        self.conv4 = self.conv_block(complexity*4, complexity*8)
        self.conv5 = self.conv_block(complexity*8, complexity*16)
        
        # 平展特征图并通过线性层输出真假值
        self.fc = nn.Linear(complexity * 16 * 18 * 18, 1)  # 线性层将特征展平为1个输出
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, input_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)  # 第一层卷积
        x = self.conv2(x)  # 第二层卷积
        x = self.conv3(x)  # 第三层卷积
        x = self.conv4(x)  # 第四层卷积
        x = self.conv5(x)  # 第五层卷积
        
        # 展平特征图为一维向量
        x = x.view(x.size(0), -1)  # (batch_size, complexity * 16 * 18 * 18)

        # 通过线性层和sigmoid输出真假值
        x = self.fc(x)  # (batch_size, 1)
        x = self.sigmoid(x)  # 使用sigmoid将输出限制在[0, 1]范围
        
        return x.view(-1)  # 输出维度为 (batch_size,)，每个值表示真假


# 使用示例
def main():
    # 创建不同复杂度的模型实例
    model_low = UNet(output_channels=2, complexity=8)
    model_medium = UNet(output_channels=2, complexity=16)
    model_high = UNet(output_channels=2, complexity=32)
    
    discriminator = Discriminator(complexity=8)
    
    # 生成一个随机的576x576的单通道图像作为输入
    input_image = torch.randn(5, 1, 576, 576)
    
    # 对每个模型进行前向传播并打印参数量
    for name, model in [("Low", model_low), ("Medium", model_medium), ("High", model_high)]:
        output = model(input_image)
        print(f"{name} complexity model:")
        print(f"  Input shape: {input_image.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Total parameters: {count_parameters(model):,}")
        print()
        output = discriminator(torch.cat([input_image, output], dim=1))
        print(f"  Discriminator output shape: {output.shape}")
        print(f"  Total parameters: {count_parameters(discriminator):,}")
        print()
        

if __name__ == "__main__":
    main()