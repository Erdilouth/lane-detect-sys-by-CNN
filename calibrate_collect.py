import cv2
import os

# --- 配置参数 ---
SAVE_DIR = "calibration_images"
CHESSBOARD_SIZE = (9, 6)  # 棋盘格内角点数量 (行, 列)
# ----------------

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Jetson Orin Nano 通常使用 CSI 或 USB 摄像头
# 如果是 USB 摄像头通常是 0，如果是 CSI 可能需要 GStreamer 管道
cap = cv2.VideoCapture(0)

print("按下 's' 保存图像，按下 'q' 退出。建议从不同角度拍摄 20 张左右。")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    # 实时寻找角点用于预览，确保拍摄时棋盘格清晰可见
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_find, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret_find:
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_find)

    cv2.imshow("Calibration Collection", display_frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        img_path = os.path.join(SAVE_DIR, f"calib_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        count += 1
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()