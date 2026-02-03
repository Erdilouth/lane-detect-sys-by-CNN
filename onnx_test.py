import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession("mobilenet_lanenet.onnx")

# 准备一个符合输入形状的随机数组
dummy_input = np.random.randn(1, 3, 256, 512).astype(np.float32)

# 跑一次推理
outputs = session.run(None, {"input": dummy_input})

print("推理成功！输出形状为:", outputs[0].shape)