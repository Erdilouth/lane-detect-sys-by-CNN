import cv2
import numpy as np
import tensorrt as trt
try:
    from cuda import cuda, cudart
except ImportError:
    # Èç¹ûÖ±½Óµ¼ÈëÊ§°Ü£¬³¢ÊÔµ¼Èë×ÓÄ£¿é£¨Õë¶Ô namespace °ü½á¹¹£©
    import cuda.cuda as cuda
    import cuda.cudart as cudart
import time

# --- ÅäÖÃ²¿·Ö ---
ENGINE_PATH = "mobilenet_lanenet.engine"
INPUT_SHAPE = (1, 3, 256, 512) 
CAMERA_ID = 0 

class LaneDetector:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape("input", INPUT_SHAPE)
        
        # ·ÖÅäÏÔ´æ
        self.in_size = trt.volume(INPUT_SHAPE) * 4 
        self.out_size = trt.volume(self.context.get_tensor_shape("output")) * 4
        
        # CUDA 12.x Ð´·¨: ·µ»ØÖµÖ±½Ó¾ÍÊÇÖ¸Õë£¬²»ÐèÒª½â°ü (err, ptr)
        # ×¢Òâ£ºcuda-python 12.x µÄÐÐÎª¿ÉÄÜÒòÐ¡°æ±¾¶øÒì
        # ÎÈÍ×Ð´·¨£º¼ì²é·µ»ØÖµÀàÐÍ
        
        err, self.d_input = cudart.cudaMalloc(self.in_size)
        err, self.d_output = cudart.cudaMalloc(self.out_size)
        
        self.h_output = np.empty(self.context.get_tensor_shape("output"), dtype=np.float32)

    def infer(self, img):
        # Ô¤´¦Àí
        resized = cv2.resize(img, (INPUT_SHAPE[3], INPUT_SHAPE[2]))
        input_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = input_data.astype(np.float32) / 255.0
        input_data = input_data.transpose(2, 0, 1)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.ascontiguousarray(input_data)

        # H2D
        cudart.cudaMemcpy(self.d_input, input_data.ctypes.data, self.in_size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        
        # Execute
        self.context.set_tensor_address("input", self.d_input)
        self.context.set_tensor_address("output", self.d_output)
        self.context.execute_v3(0)
        
        # D2H
        cudart.cudaMemcpy(self.h_output.ctypes.data, self.d_output, self.out_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        
        return self.h_output

    def visualize(self, frame, output_mask):
        mask = output_mask[0, 0]
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        color_mask = np.zeros_like(frame)
        color_mask[mask_resized > 0] = [0, 255, 0] 
        return cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

def main():
    # ³õÊ¼»¯
    err, = cuda.cuInit(0)
    if err != cuda.CUresult.CUDA_SUCCESS:
        print(f"CUDA ³õÊ¼»¯Ê§°Ü: {err}")
        return

    print("ÕýÔÚ¼ÓÔØ TensorRT ÒýÇæ...")
    try:
        detector = LaneDetector(ENGINE_PATH)
    except Exception as e:
        print(f"ÒýÇæ¼ÓÔØÊ§°Ü: {e}")
        return
    print("Ä£ÐÍ¼ÓÔØÍê³É£¡")

    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("ÎÞ·¨´ò¿ªÉãÏñÍ·£¡")
        return

    fps_count = 0
    fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        start_time = time.time()
        output = detector.infer(frame)
        result_frame = detector.visualize(frame, output)
        
        fps_count += 1
        if time.time() - fps_start > 1.0:
            fps = fps_count / (time.time() - fps_start)
            print(f"FPS: {fps:.1f}")
            fps_count = 0
            fps_start = time.time()

        cv2.putText(result_frame, f"Jetson Orin - FPS: {1.0/(time.time()-start_time):.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Lane Detection", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
