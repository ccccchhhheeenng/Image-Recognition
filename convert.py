import cv2
import numpy as np

# 讀取灰階圖像
shapes = ['circle','triangle',  'cube', 'cross']
for shape in shapes:
    x = 1
    while x <= 1000:
        input_path = f"{shape}/image/{shape}{x}.png"
        output_path = f"{shape}/grayscale/{shape}{x}.txt"
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        image = np.array(fix_img) / 255.0  
        np.savetxt(output_path, image, delimiter=',', fmt='%.6f')
        x += 1
# x = 1
# while x <= 4:
#     input_path = f"test_image/image{x}.png"
#     output_path = f"test_image/image{x}.txt"
#     img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
#     fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
#     image = np.array(fix_img) / 255.0  
#     np.savetxt(output_path, image, delimiter=',', fmt='%.6f')
#     x += 1
print("=========FINISH========")