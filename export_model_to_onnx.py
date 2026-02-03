import torch
import torch.onnx
from model import MobileNetLaneNet

def export_to_onnx(model_path="mobilenet_lanenet.onnx"):
    # 1. 初始化模型并加载权重 (如果有的话)
    model = MobileNetLaneNet(out_channels=1)
    model.load_state_dict(torch.load("lane_model_epoch_10.pth")) # 如果有权重请取消注释

    # 2. 切换到评估模式 (非常重要！会影响 BatchNorm 和 Dropout)
    model.eval()

    # 3. 定义虚拟输入 (Dummy Input)
    # 形状必须与模型预期一致: (Batch_size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 256, 512)

    # 4. 执行导出
    print(f"正在转换模型到 ONNX...")
    torch.onnx.export(
        model,  # 要导出的模型
        dummy_input,  # 虚拟输入
        model_path,  # 保存路径
        export_params=True,  # 导出训练后的参数权重
        opset_version=12,  # ONNX 算子版本 (11及以上支持双线性上采样)
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],  # 输入节点名称
        output_names=['output'],  # 输出节点名称
        # 设置动态维度，方便后续推理时更改 Batch Size 或分辨率
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"导出成功！模型已保存至: {model_path}")


if __name__ == "__main__":
    export_to_onnx()