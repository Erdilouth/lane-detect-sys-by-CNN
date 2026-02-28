import cv2
import numpy as np
import tensorrt as trt
from cuda import cuda, cudart  # ÊÊÅä JetPack 6.2 µÄµ¼Èë·½Ê½
import time
import os
import sys

# --- ÅäÖÃ²¿·Ö ---
# ÇëÐÞ¸ÄÎªÄãµÄÊÓÆµÎÄ¼þÂ·¾¶
VIDEO_PATH = "/home/erdilouth/test_video.mp4" 
ENGINE_PATH = "mobilenet_lanenet.engine"
INPUT_SHAPE = (1, 3, 256, 512)  # (Batch, Channel, Height, Width)

class LaneDetector:
    def __init__(self, engine_path):
        # 1. ³õÊ¼»¯ CUDA Çý¶¯ (·Ç³£ÖØÒª)
        err, = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA ³õÊ¼»¯Ê§°Ü: {err}")

        # 2. ¼ÓÔØ TensorRT ÒýÇæ
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"ÕÒ²»µ½ÒýÇæÎÄ¼þ: {engine_path}")

        print(f"ÕýÔÚ¼ÓÔØÒýÇæ: {engine_path} ...")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # ´¦Àí¶¯Ì¬ÊäÈëÐÎ×´ (Õë¶Ô TensorRT 10.x)
        self.context.set_input_shape("input", INPUT_SHAPE)
        
        # 3. ·ÖÅäÏÔ´æ
        self.in_size = trt.volume(INPUT_SHAPE) * 4 # float32 = 4 bytes
        output_shape = self.context.get_tensor_shape("output")
        self.out_size = trt.volume(output_shape) * 4
        
        # CUDA 12.x ±ê×¼·ÖÅä·½Ê½
        err, self.d_input = cudart.cudaMalloc(self.in_size)
        err, self.d_output = cudart.cudaMalloc(self.out_size)
        
        # Ô¤·ÖÅä Host ÄÚ´æ (ËøÒ³ÄÚ´æ»á¸ü¿ì£¬µ«ÕâÀïÎªÁË¼òµ¥ÓÃÆÕÍ¨ numpy)
        self.h_output = np.empty(output_shape, dtype=np.float32)
        
        # ´´½¨ CUDA Á÷ (Stream) ÒÔÊµÏÖÒì²½²Ù×÷ (¿ÉÑ¡£¬ÕâÀïÓÃÄ¬ÈÏÁ÷ 0)
        self.stream = 0 

    def infer(self, img):
        # --- Ô¤´¦Àí (CPU) ---
        # Resize -> RGB -> Normalize -> CHW
        resized = cv2.resize(img, (INPUT_SHAPE[3], INPUT_SHAPE[2]))
        input_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = input_data.astype(np.float32) / 255.0
        input_data = input_data.transpose(2, 0, 1) # HWC -> CHW
        input_data = np.expand_dims(input_data, axis=0) # Add Batch dim
        input_data = np.ascontiguousarray(input_data)

        # --- ÍÆÀí (GPU) ---
        # 1. ¿½±´Êý¾Ý Host -> Device
        # ×¢Òâ£ºÊ¹ÓÃ cudart.cudaMemcpy ¶ø²»ÊÇ runtime.cudaMemcpy
        cudart.cudaMemcpy(self.d_input, input_data.ctypes.data, self.in_size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        
        # 2. Ö´ÐÐÍÆÀí (ÐÞ¸´µã£ºÊ¹ÓÃ execute_v2)
        # execute_v2 ÐèÒª´«ÈëÏÔ´æµØÖ·ÁÐ±í [input_ptr, output_ptr]
        # ×¢Òâ£ºÕâÀïµÄË³Ðò±ØÐëºÍÄ£ÐÍ°ó¶¨µÄË³ÐòÒ»ÖÂ (Í¨³£ÊÇ input, output)
        bindings = [int(self.d_input), int(self.d_output)]
        
        self.context.execute_v2(bindings=bindings)
        
        # 3. ¿½±´½á¹û Device -> Host
        cudart.cudaMemcpy(self.h_output.ctypes.data, self.d_output, self.out_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        
        return self.h_output

    def visualize(self, frame, output_mask):
        # --- ºó´¦Àí (CPU) ---
        mask = output_mask[0, 0]
        # ¶þÖµ»¯
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # µ÷Õû Mask ´óÐ¡ÒÔÆ¥ÅäÔ­ÊÓÆµÖ¡
        mask_resized = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # ´´½¨ÂÌÉ«¸²¸Ç²ã
        color_mask = np.zeros_like(frame)
        color_mask[mask_resized > 0] = [0, 255, 0] # BGR ÂÌÉ«
        
        # µþ¼Ó
        return cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

    def cleanup(self):
        # ÊÍ·ÅÏÔ´æ
        cudart.cudaFree(self.d_input)
        cudart.cudaFree(self.d_output)

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"´íÎó£ºÕÒ²»µ½ÊÓÆµÎÄ¼þ {VIDEO_PATH}")
        print("ÇëÉÏ´«Ò»¸ö mp4 ÎÄ¼þ²¢ÔÚ´úÂëÖÐÐÞ¸Ä VIDEO_PATH")
        return

    try:
        detector = LaneDetector(ENGINE_PATH)
        print("? Ä£ÐÍ¼ÓÔØ³É¹¦£¡")
    except Exception as e:
        print(f"? Ä£ÐÍ¼ÓÔØÊ§°Ü: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ÎÞ·¨´ò¿ªÊÓÆµÎÄ¼þ£¡")
        return

    fps_count = 0
    fps_start = time.time()
    
    print("¿ªÊ¼ÍÆÀí... °´ 'q' ÍË³ö")

    while True:
        ret, frame = cap.read()
        
        # ÊÓÆµ²¥·ÅÍêºó×Ô¶¯ÖØ²¥
        if not ret:
            print("ÊÓÆµ²¥·Å½áÊø£¬ÖØÐÂ¿ªÊ¼...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # ¼ÇÂ¼ÍÆÀí¿ªÊ¼Ê±¼ä
        t0 = time.time()
        
        # Ö´ÐÐÍÆÀí
        output = detector.infer(frame)
        
        # Ö´ÐÐ¿ÉÊÓ»¯
        result_frame = detector.visualize(frame, output)
        
        # ¼ÆËã FPS
        t1 = time.time()
        fps_count += 1
        if t1 - fps_start > 1.0:
            fps = fps_count / (t1 - fps_start)
            print(f"µ±Ç° FPS: {fps:.1f} (ÍÆÀí+´¦ÀíÑÓ³Ù: {(t1-t0)*1000:.1f}ms)")
            fps_count = 0
            fps_start = t1

        # ÔÚ»­ÃæÉÏÏÔÊ¾ FPS
        cv2.putText(result_frame, f"Orin Nano FPS: {1.0/(t1-t0+1e-5):.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Lane Detection (Video Test)", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.cleanup()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
