import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk

from model import MobileNetLaneNet


class LaneDetectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("基于CNN的车道线检测系统")
        self.root.geometry("1000x600")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.img_size = (512, 256)

        self.build_ui()

    def build_ui(self):
        # ===== 左侧容器（重点）=====
        left_frame = tk.Frame(self.root, bg="black")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.image_label = tk.Label(
            left_frame,
            bg="black",
            text="图像显示区域",
            fg="white"
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # ===== 右侧按钮区 =====
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=20)

        tk.Button(btn_frame, text="加载模型", width=18, command=self.load_model).pack(pady=10)
        tk.Button(btn_frame, text="选择图片检测", width=18, command=self.detect_image).pack(pady=10)
        tk.Button(btn_frame, text="退出系统", width=18, command=self.root.quit).pack(pady=10)

    # ================= 功能函数 =================

    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="选择模型权重",
            filetypes=[("PyTorch Model", "*.pth")]
        )
        if not model_path:
            return

        self.model = MobileNetLaneNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        messagebox.showinfo("提示", "模型加载成功！")

    def detect_image(self):
        if self.model is None:
            messagebox.showwarning("警告", "请先加载模型！")
            return

        img_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )
        if not img_path:
            return

        result_img = self.predict(img_path)
        self.show_image(result_img)

    def predict(self, img_path):
        original_img = cv2.imread(img_path)
        h, w = original_img.shape[:2]

        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.img_size)

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        mask = output.squeeze().cpu().numpy()
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)

        result = original_img.copy()
        result[mask_resized > 0] = [0, 0, 255]  # 红色车道线

        return result

    def show_image(self, img_bgr):
        self.root.update_idletasks()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        label_w = self.image_label.winfo_width()
        label_h = self.image_label.winfo_height()

        img_w, img_h = img_pil.size
        scale = min(label_w / img_w, label_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = LaneDetectApp(root)
    root.mainloop()
