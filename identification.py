import numpy as np
import cv2

img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
triangle = np.array(fix_img) / 255.0  # 正規化
np.savetxt('image.txt', triangle, delimiter=',', fmt='%1d')
# 載入 image

image = np.loadtxt("image.txt", delimiter=',').flatten().reshape(1, -1)  # reshape 使其成為 2D
a=["heart","star","circle","triangle"]
# 載入模型參數
weights_input_hidden = np.loadtxt("weights_input_hidden.txt", delimiter=",")
weights_hidden_output = np.loadtxt("weights_hidden_output.txt", delimiter=",")
bias_hidden = np.loadtxt("bias_hidden.txt", delimiter=",").reshape(1, -1)
bias_output = np.loadtxt("bias_output.txt", delimiter=",").reshape(1, -1)



# 同樣的前向傳遞步驟
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

hidden_input = np.dot(image, weights_input_hidden) + bias_hidden
hidden_output = sigmoid(hidden_input)
final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
final_output = softmax(final_input)

predicted_class = np.argmax(final_output)
print(final_output)

print("\n預測類別：", a[predicted_class])
