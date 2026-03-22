import cv2
import numpy as np
import glob
import json

# --- 配置参数 ---
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 18.0  # 棋盘格每个小方块的实际尺寸 (mm)
IMAGE_PATH = "calibration_images/*.jpg"
# ----------------

# 准备对象点 (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 真实世界中的 3D 点
imgpoints = []  # 图像平面中的 2D 点

images = glob.glob(IMAGE_PATH)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)
        # 精细化角点坐标
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("标定成功！")
    print(f"相机内参 (Camera Matrix):\n{mtx}")
    print(f"畸变系数 (Distortion Coeffs):\n{dist}")

    # 保存参数供后续 IPM 使用
    calib_data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeff": dist.tolist()
    }
    with open("camera_params.json", "w") as f:
        json.dump(calib_data, f)
    print("参数已保存至 camera_params.json")