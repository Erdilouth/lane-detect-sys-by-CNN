import os
from cuda.bindings import driver, runtime

# 1. Windows 环境下依然建议加上这行
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)


def final_test():
    try:
        # 使用 driver 替代之前的 cuda
        res, = driver.cuInit(0)
        if res == driver.CUresult.CUDA_SUCCESS:
            print("✅ CUDA Driver API (driver) 成功初始化！")

            res, dev_count = driver.cuDeviceGetCount()
            print(f"✅ 检测到 {dev_count} 个 GPU 设备")

        # 使用 runtime 替代之前的 cudart
        res, = runtime.cudaFree(0)
        if res == runtime.cudaError_t.cudaSuccess:
            print("✅ CUDA Runtime API (runtime) 成功初始化！")

    except Exception as e:
        print(f"❌ 依然报错: {e}")
        print("提示：请检查是否安装了多个版本的 cuda-python 导致冲突。")


if __name__ == "__main__":
    final_test()