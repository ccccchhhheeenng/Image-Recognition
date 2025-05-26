import numpy as np
import time
import os
# 載入形狀影像資料
def load_shape_matrix(file):
    return np.loadtxt(file, delimiter=',')

# 載入輸入影像（要預測的影像）
image = np.loadtxt("image.txt", delimiter=',').flatten()

# 激活函數與其導數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Softmax 函數（多類別輸出）
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # 避免 overflow
    return exps / np.sum(exps, axis=1, keepdims=True)

# 交叉熵損失函數
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

shapes = ['circle', 'cross', 'cube', 'triangle']
subdir = 'grayscale'  # 假設所有形狀都在 "grayscale" 這個子目錄內
num_shapes = 300

# 讀取所有形狀的矩陣並展平
inputs = np.array([
    load_shape_matrix(os.path.join(shape, subdir, f'{shape}{i}.txt')).flatten()
    for shape in shapes
    for i in range(1, num_shapes + 1)
])

# 檢查輸出格式
outputs = np.array([
    [1 if j == shapes.index(shape) else 0 for j in range(len(shapes))]
    for shape in shapes
    for _ in range(num_shapes)
])

# 網路參數
np.random.seed(int(time.time()))
input_layer_size = 4096
hidden_layer_size = 256

output_layer_size = 4

# 初始化權重與偏差
weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size) * np.sqrt(1. / input_layer_size)
weights_hidden_output = np.random.randn(hidden_layer_size, output_layer_size) * np.sqrt(1. / hidden_layer_size)
bias_hidden = np.random.uniform(size=(1, hidden_layer_size))
bias_output = np.random.uniform(size=(1, output_layer_size))

# 訓練參數
learning_rate = 0.0001
epochs = 2000

# 訓練過程
for epoch in range(epochs):
    # 前向傳遞
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = softmax(final_input)

    # 損失計算
    loss = cross_entropy(outputs, final_output)

    # 反向傳遞（softmax + cross entropy 梯度簡化）
    d_output = final_output - outputs
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # 更新權重與偏差
    weights_hidden_output -= hidden_output.T.dot(d_output) * learning_rate
    bias_output -= np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden -= inputs.T.dot(d_hidden) * learning_rate
    bias_hidden -= np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # 列印訓練進度
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# === 預測輸入影像分類 ===
hidden_input = np.dot(image, weights_input_hidden) + bias_hidden
hidden_output = sigmoid(hidden_input)
final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
final_output = softmax(final_input)

# 預測類別
# 儲存模型參數
np.savetxt("weights_input_hidden.txt", weights_input_hidden, delimiter=",")
np.savetxt("weights_hidden_output.txt", weights_hidden_output, delimiter=",")
np.savetxt("bias_hidden.txt", bias_hidden, delimiter=",")
np.savetxt("bias_output.txt", bias_output, delimiter=",")

predicted_class = np.argmax(final_output)
print(final_output)
print("\n預測類別：", predicted_class)
