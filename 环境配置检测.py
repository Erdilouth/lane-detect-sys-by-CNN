import torch
import cv2
print(cv2.__version__) #输出版本
torch.set_default_device('cuda')
print(torch.__version__)  # 输出版本，如2.1.0
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("未检测到GPU，使用CPU版本")