import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 讀取矩陣函式
def load_shape_matrix(file):
    return np.loadtxt(file, delimiter=',')


# 讀取訓練數據
X_train = np.array([
    load_shape_matrix('diamond1.txt').flatten(),
    load_shape_matrix('circle1.txt').flatten(),
    load_shape_matrix('cube1.txt').flatten(),
    load_shape_matrix('triangle1.txt').flatten()
])

# 標籤 (0:菱形, 1:圓形, 2:正方形, 3:三角形)
y_train = np.array([0, 1, 2, 3])

# 轉換為 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# 定義神經網路
class ShapeClassifier(nn.Module):
    def __init__(self):
        super(ShapeClassifier, self).__init__()
        self.fc1 = nn.Linear(160*160, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型
model = ShapeClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
def train_model(epochs=500):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# 測試 `image.png` 的形狀
def predict_shape(image_file):
    image_matrix = load_shape_matrix(image_file).flatten()
    image_tensor = torch.tensor(image_matrix, dtype=torch.float32)

    model.eval()
    output = model(image_tensor)
    predicted_shape = torch.argmax(output).item()

    shapes = ["Diamond", "Circle", "Square", "Triangle"]
    return shapes[predicted_shape]

# 執行訓練
train_model()

# 預測 `image.png`
shape_result = predict_shape('image.txt')
print(f'Predicted Shape: {shape_result}')