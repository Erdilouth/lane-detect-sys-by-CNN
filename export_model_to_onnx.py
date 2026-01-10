import torch
from torchvision import models
import torch.nn as nn
import onnx
from tqdm import tqdm
from model import MobileNetLaneNet

def export_model_to_onnx(model_path, output_onnx_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # 加载预训练模型
    model = torch.load(model_path)
    model = model.eval().to(device)

    # 构建一个输入张量（与已训练模型保持一致）
    x = torch.rand(1, 3, 256, 512).to(device)
    print(x.shape)

    # Perform inference with a progress bar
    with torch.no_grad():
        pbar = tqdm(total=1, desc='Exporting ONNX')  # 进度条
        torch.onnx.export(
            model,                    # Model to export
            x,                        # Example input
            output_onnx_path,         # Output ONNX file name
            input_names=['input'],    # Input names (can be customized)
            output_names=['output'],  # Output names (can be customized)
            opset_version=11,         # ONNX operator set version
        )
        pbar.update(1)
        pbar.close()

    # Validate the exported ONNX model
    onnx_model = onnx.load(output_onnx_path)
    # Check if the model format is correct
    onnx.checker.check_model(onnx_model)
    print('无报错，onnx模型导入成功!')

if __name__ == '__main__':
    # 调用函数进行模型导出
    model_path = 'lane_model_epoch_10.pth'
    output_onnx_path = 'Lane_detect_sys_by_CNN.onnx'
    export_model_to_onnx(model_path, output_onnx_path)
