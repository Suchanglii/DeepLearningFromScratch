import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

from ch05.two_layer_net import TwoLayerNet
from ch07.simple_convnet import SimpleConvNet
from common.multi_layer_net import MultiLayerNet


class DrawingBoard:
    def __init__(self, root,network):
        self.network = network
        self.root = root
        self.root.title("MNIST 手写测试板")

        # 定义参数
        self.canvas_size = 280  # 窗口显示大小
        self.output_size = 28  # 神经网络需要的输入大小
        self.bg_color = "black"
        self.fg_color = "white"
        self.line_width = 15  # 画笔粗细

        # 创建画布
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg=self.bg_color)
        self.canvas.pack(pady=10)

        # 创建一个内存中的图片对象，用于后端处理
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)

        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)

        # 按钮控制
        self.btn_clear = tk.Button(root, text="清空画布", command=self.clear_canvas)
        self.btn_clear.pack(side=tk.LEFT, padx=20, pady=10)

        self.btn_get = tk.Button(root, text="提取 28x28 数据", command=self.get_data)
        self.btn_get.pack(side=tk.RIGHT, padx=20, pady=10)

    def paint(self, event):
        # 获取坐标并绘制
        x1, y1 = (event.x - self.line_width), (event.y - self.line_width)
        x2, y2 = (event.x + self.line_width), (event.y + self.line_width)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.fg_color, outline=self.fg_color)
        self.draw.ellipse([x1, y1, x2, y2], fill=self.fg_color)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)

    def get_data(self):
        # 1. 缩放到 28x28
        img_resized = self.image.resize((self.output_size, self.output_size), Image.LANCZOS)
        # 2. 转为 numpy 数组
        data = np.array(img_resized)
        # 3. 归一化 (0-1)
        data = data / 255.0

        # print(f"数据已生成! 形状: {data.shape}")
        # print(data)  # 打印出 28x28 的矩阵

        # 这里你可以直接接入你的模型预测逻辑：
        if not isinstance(network,SimpleConvNet):
            # 普通神经网
            prediction = network.predict(data.reshape(1, 784))
        else:
            data = data.reshape(1, 1, 28, 28)
            # 或者
            # data = np.expand_dims(data, axis=0)
            prediction = network.predict(data)
        print("预测结果:", np.argmax(prediction))

        return data


if __name__ == "__main__":
    # 模型初始化
    # network = MultiLayerNet(
    #     input_size=784, hidden_size_list=[100, 100, 100, 100],
    #     output_size=10)
    network = SimpleConvNet(input_dim=(1, 28, 28),
                            conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                            hidden_size=100, output_size=10, weight_init_std=0.01)
    network.load_params(file_name="D:\\Documents\\DeepLearningFromScratch\\ch07\\params1769741000.pkl")
    # weight = dict(np.load("model_weights.npz"))
    # network.load_params(weight)

    root = tk.Tk()
    app = DrawingBoard(root,network)
    root.mainloop()
