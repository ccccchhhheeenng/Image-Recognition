import cv2
import numpy as np

# 讀取灰階圖像
#菱形
img = cv2.imread('heart1.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
heart = np.array(fix_img) / 255.0  # 正規化
np.savetxt('heart1.txt', heart, delimiter=',', fmt='%1d')

img = cv2.imread('heart2.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
heart2 = np.array(fix_img) / 255.0  # 正規化
np.savetxt('heart2.txt', heart2, delimiter=',', fmt='%1d')

img = cv2.imread('heart3.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
heart2 = np.array(fix_img) / 255.0  # 正規化
np.savetxt('heart3.txt', heart2, delimiter=',', fmt='%1d')

#圓形
img = cv2.imread('star1.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
star = np.array(fix_img) / 255.0  # 正規化
np.savetxt('star1.txt', star, delimiter=',', fmt='%1d')

img = cv2.imread('star2.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
star = np.array(fix_img) / 255.0  # 正規化
np.savetxt('star2.txt', star, delimiter=',', fmt='%1d')

img = cv2.imread('star3.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
star = np.array(fix_img) / 255.0  # 正規化
np.savetxt('star3.txt', star, delimiter=',', fmt='%1d')

#正方形
img = cv2.imread('circle1.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
circle = np.array(fix_img) / 255.0  # 正規化
np.savetxt('circle1.txt', circle, delimiter=',', fmt='%1d')

img = cv2.imread('circle2.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
circle = np.array(fix_img) / 255.0  # 正規化
np.savetxt('circle2.txt', circle, delimiter=',', fmt='%1d')

img = cv2.imread('circle3.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
circle = np.array(fix_img) / 255.0  # 正規化
np.savetxt('circle3.txt', circle, delimiter=',', fmt='%1d')

#三角形
img = cv2.imread('triangle1.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
triangle = np.array(fix_img) / 255.0  # 正規化
np.savetxt('triangle1.txt', triangle, delimiter=',', fmt='%1d')

img = cv2.imread('triangle2.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
triangle = np.array(fix_img) / 255.0  # 正規化
np.savetxt('triangle2.txt', triangle, delimiter=',', fmt='%1d')

img = cv2.imread('triangle3.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
triangle = np.array(fix_img) / 255.0  # 正規化
np.savetxt('triangle3.txt', triangle, delimiter=',', fmt='%1d')


img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
fix_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
triangle = np.array(fix_img) / 255.0  # 正規化
np.savetxt('image.txt', triangle, delimiter=',', fmt='%1d')
