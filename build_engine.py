import tensorrt as trt
import os


def build_engine(onnx_file_path, engine_file_path):
    # 将日志等级改为 VERBOSE 或 INFO，以便看到更多细节
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)

    # 1. 定义网络
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 2. 解析 ONNX
    if not os.path.exists(onnx_file_path):
        print(f"错误: 找不到文件 {onnx_file_path}")
        return

    print(f"正在解析 ONNX 文件: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: ONNX 解析失败!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

        # 3. 配置转换参数
        config = builder.create_builder_config()

        # --- 新增：处理动态输入 ---
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name  # 获取输入节点的名称，通常是 "input"

        # 设置动态维度的范围: [min, opt, max]
        # 我们之前的输入是 (1, 3, 256, 512)
        min_shape = (1, 3, 256, 512)
        opt_shape = (1, 3, 256, 512)
        max_shape = (4, 3, 256, 512)  # 允许最大 Batch 为 4，尺寸也可以调大

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        # -----------------------

        # 设置最大临时显存
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        if builder.platform_has_fast_fp16:
            print("硬件支持 FP16，已开启加速。")
            config.set_flag(trt.BuilderFlag.FP16)

    if builder.platform_has_fast_fp16:
        print("硬件支持 FP16，已开启加速。")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("警告: 硬件不支持 FP16，将使用 FP32。")

    # 4. 构建序列化引擎
    print("正在构建 Engine (这一步会非常慢，请耐心等待)...")
    serialized_engine = builder.build_serialized_network(network, config)

    # 5. 检查结果并保存
    if serialized_engine is None:
        print("错误: 构建 Engine 失败！请检查上方输出的详细日志。")
        return

    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"恭喜！Engine 构建成功: {engine_file_path}")


if __name__ == "__main__":
    build_engine("mobilenet_lanenet.onnx", "mobilenet_lanenet.engine")