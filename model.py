import torch
import torch.nn as nn
from torchvision import models


# 定义轻量级上采样模块
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # 使用双倍上采样 + 卷积
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # 拼接来自 MobileNet 的跳跃连接特征
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetLaneNet(nn.Module):
    def __init__(self, out_channels=1):
        super(MobileNetLaneNet, self).__init__()

        # 1. 加载预训练的 MobileNet V2 特征提取层
        # weights=models.MobileNet_V2_Weights.DEFAULT 表示使用在大数据集上训练好的参数
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features

        # 2. 提取不同层级的特征（用于跳跃连接，恢复细节）
        self.layer0 = backbone[0:2]  # 尺寸: 1/2
        self.layer1 = backbone[2:4]  # 尺寸: 1/4
        self.layer2 = backbone[4:7]  # 尺寸: 1/8
        self.layer3 = backbone[7:14]  # 尺寸: 1/16
        self.layer4 = backbone[14:18]  # 尺寸: 1/32 (最深层)

        # 3. 构建解码器 (Decoder)
        # 注意：这里的输入通道数要根据 MobileNet V2 的输出逐一对应
        self.up1 = UpBlock(320 + 96, 96)  # 1/32 -> 1/16
        self.up2 = UpBlock(96 + 32, 32)  # 1/16 -> 1/8
        self.up3 = UpBlock(32 + 24, 24)  # 1/8 -> 1/4
        self.up4 = UpBlock(24 + 16, 16)  # 1/4 -> 1/2

        # 4. 最后一步：回到原图尺寸并输出
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码阶段 (Forward pass through MobileNetV2)
        s0 = self.layer0(x)  # 1/2
        s1 = self.layer1(s0)  # 1/4
        s2 = self.layer2(s1)  # 1/8
        s3 = self.layer3(s2)  # 1/16
        s4 = self.layer4(s3)  # 1/32

        # 解码阶段 (Upsampling)
        x = self.up1(s4, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)

        x = self.final_up(x)
        x = self.final_conv(x)
        return self.sigmoid(x)


if __name__ == "__main__":
    # 测试模型
    model = MobileNetLaneNet()
    test_input = torch.randn(1, 3, 256, 512)
    output = model(test_input)
    print(f"MobileNet V2 成功运行！输出尺寸: {output.shape}")