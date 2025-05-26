import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torch
import os

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("畫一個圖（圓形、三角形、方形、十字）")
        self.canvas = tk.Canvas(root, width=256, height=256, bg='white')
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (256, 256), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        tk.Button(root, text="辨識", command=self.predict).pack(side=tk.LEFT, padx=10)
        tk.Button(root, text="清除", command=self.clear).pack(side=tk.LEFT, padx=10)

        self.result_label = tk.Label(root, text="預測: None")
        self.result_label.pack(pady=10)
        self.prob_label = tk.Label(root, text="機率: None")
        self.prob_label.pack(pady=5)

        self.last_x, self.last_y = None, None
        self.classes = ['圓形', '三角形', '方形', '十字']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.weights_input_hidden = torch.load("weights_input_hidden.pt").to(self.device)
            self.weights_hidden_output = torch.load("weights_hidden_output.pt").to(self.device)
            self.bias_hidden = torch.load("bias_hidden.pt").to(self.device)
            self.bias_output = torch.load("bias_output.pt").to(self.device)
        except FileNotFoundError:
            print("模型參數檔案未找到，請確認 .pt 檔案存在")
            exit()

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=8, fill='black')
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=8)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (256, 256), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="預測: None")
        self.prob_label.config(text="機率: None")

    def predict(self):
        img = self.image.resize((64, 64), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = img_array.flatten()
        image_tensor = torch.tensor(img_array, dtype=torch.float32).to(self.device).unsqueeze(0)  # shape: (1, 4096)

        # 前向傳播
        hidden_input = torch.matmul(image_tensor, self.weights_input_hidden) + self.bias_hidden
        hidden_output = torch.relu(hidden_input)
        final_input = torch.matmul(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = torch.softmax(final_input, dim=1).cpu().numpy()[0]

        predicted_class = np.argmax(final_output)
        self.result_label.config(text=f"預測: {self.classes[predicted_class]}")
        self.prob_label.config(text=f"機率: {final_output}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
