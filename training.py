
import numpy as np
import torch
import os
import time

# 設定GPU為優先運算設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("運算設備：", device)

# 讀資料的function
def load_shape_matrix(file):
    return np.loadtxt(file, delimiter=',')

# 定義形狀與資料夾
shapes = ['circle', 'cross', 'cube', 'triangle']
subdir = 'grayscale'
num_shapes = 1000

# 載入要訓練的照片
inputs = np.array([
load_shape_matrix(os.path.join(shape, subdir, f'{shape}{i}.txt')).flatten()
for shape in shapes
for i in range(1, num_shapes + 1)
], dtype=np.float32)

#初始化輸出層
labels = np.array([
shapes.index(shape)
for shape in shapes
for _ in range(num_shapes)
], dtype=np.int64)

# 轉為 Tensor 並移至裝置準備運算
tensor_inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
tensor_labels = torch.tensor(labels, dtype=torch.long).to(device)

# 設定隱藏層大小和輸入、輸出層大小
np.random.seed(int(time.time()))
input_size = 4096
hidden_size = 512
output_size = 4

# 初始化隱藏層和輸出層的權重與偏差
weights_input_hidden = torch.randn(input_size, hidden_size, dtype=torch.float32) * np.sqrt(1. / input_size)
weights_hidden_output = torch.randn(hidden_size, output_size, dtype=torch.float32) * np.sqrt(1. / hidden_size)
bias_hidden = torch.rand(1, hidden_size, dtype=torch.float32)
bias_output = torch.rand(1, output_size, dtype=torch.float32)

# 將隱藏層的矩陣和輸出層的矩陣移至裝置
weights_input_hidden = weights_input_hidden.to(device)
weights_hidden_output = weights_hidden_output.to(device)
bias_hidden = bias_hidden.to(device)
bias_output = bias_output.to(device)

# 訓練參數(次數、學習率)
learning_rate = 0.0015
epochs = 1000000

# 訓練過程
for epoch in range(epochs):
    # 前向傳播
    hidden_input = torch.matmul(tensor_inputs, weights_input_hidden) + bias_hidden
    hidden_output = torch.relu(hidden_input)
    final_input = torch.matmul(hidden_output, weights_hidden_output) + bias_output
    final_output = torch.nn.functional.log_softmax(final_input, dim=1)
    
    # 計算損失與反向傳遞
    loss = torch.nn.functional.nll_loss(final_output, tensor_labels)
    probs = torch.exp(final_output)
    d_output = probs
    d_output[range(len(tensor_labels)), tensor_labels] -= 1
    d_output /= len(tensor_labels)

    # 反向傳遞
    grad_weights_hidden_output = torch.matmul(hidden_output.T, d_output)
    grad_bias_output = torch.sum(d_output, dim=0, keepdim=True)

    d_hidden_linear = torch.matmul(d_output, weights_hidden_output.T)
    d_hidden = d_hidden_linear * (hidden_output > 0).float()

    grad_weights_input_hidden = torch.matmul(tensor_inputs.T, d_hidden)
    grad_bias_hidden = torch.sum(d_hidden, dim=0, keepdim=True)

    # 更新權重與偏差
    weights_input_hidden -= learning_rate * grad_weights_input_hidden
    weights_hidden_output -= learning_rate * grad_weights_hidden_output
    bias_hidden -= learning_rate * grad_bias_hidden
    bias_output -= learning_rate * grad_bias_output

    # 每50次輸出一次損失
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 運算結束，儲存模型參數
torch.save(weights_input_hidden.cpu(), "weights_input_hidden.pt")
torch.save(weights_hidden_output.cpu(), "weights_hidden_output.pt")
torch.save(bias_hidden.cpu(), "bias_hidden.pt")
torch.save(bias_output.cpu(), "bias_output.pt")