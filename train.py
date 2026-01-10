import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CULaneDataset  # 导入你之前的类
from model import MobileNetLaneNet  # 导入你之前的模型

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs: 模型的输出 (经过 Sigmoid)
        # targets: 真实的标签
        
        # 展平 tensor，方便计算交集
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def train():
    # 1. 配置参数 (Hyperparameters)
    data_root = 'D:/database/archive/CULane/driver_161_90frame/driver_161_90frame'  # 数据集路径
    train_list = 'train_list.txt'
    val_list = 'val_list.txt'

    batch_size = 8  # 每次喂给模型多少张图（如果显存报错，减小到 4 或 2）
    epochs = 10  # 全部数据训练多少轮
    learning_rate = 1e-4  # 学习率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用GPU

    # 2. 准备数据加载器 (DataLoader)
    train_ds = CULaneDataset(data_root, train_list)
    val_ds = CULaneDataset(data_root, val_list)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. 初始化模型、损失函数和优化器
    model = MobileNetLaneNet().to(device)
    # DICEloss: 损失函数（车道线 vs 背景）
    criterion = DiceLoss()
    # Adam: 自动调整学习率的优化器，非常适合新手
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"开始在 {device} 上训练...")

    # 4. 训练循环
    for epoch in range(epochs):
        model.train()  # 切换到训练模式
        running_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            # 将数据搬到 GPU/CPU
            images = images.to(device)
            masks = masks.to(device).float().unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

            # --- 核心三步 ---
            optimizer.zero_grad()  # 1. 清空上次的梯度
            outputs = model(images)  # 2. 模型预测 (前向传播)
            loss = criterion(outputs, masks)  # 3. 计算误差
            loss.backward()  # 4. 反向传播（算出误差对参数的影响）
            optimizer.step()  # 5. 更新参数

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 每个 Epoch 结束后保存模型权重
        torch.save(model.state_dict(), f'lane_model_epoch_{epoch + 1}.pth')
        print(f"第 {epoch + 1} 轮训练完成，权重已保存。")


if __name__ == "__main__":
    train()
