import pygame
import random
import asyncio
import platform
import os
import math

# 初始化Pygame
pygame.init()

# 設定參數
WIDTH, HEIGHT = 500, 500
OUTPUT_DIR = "triangle/image"
NUM_IMAGES = 1000

# 確保輸出目錄存在（模擬檔案系統，Pyodide環境下實際由瀏覽器處理）
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except:
    pass

def rotate_point(x, y, angle, pivot_x, pivot_y):
    # 圍繞指定點旋轉座標
    rad = math.radians(angle)
    new_x = pivot_x + (x - pivot_x) * math.cos(rad) - (y - pivot_y) * math.sin(rad)
    new_y = pivot_y + (x - pivot_x) * math.sin(rad) + (y - pivot_y) * math.cos(rad)
    return (new_x, new_y)

def generate_hollow_rotated_triangle_image(index):
    # 創建表面
    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill((255, 255, 255))  # 白色背景
    
    # 隨機三角形參數
    size = random.randint(130, 288)  # 三角形邊長範圍50-288像素
    # 調整中心點範圍，確保三角形不超出邊界
    max_offset = size * (3 ** 0.5) / 2  # 正三角形最大高度
    center_x = random.randint(int(max_offset), WIDTH - int(max_offset))
    center_y = random.randint(int(max_offset), HEIGHT - int(max_offset))
    angle = random.randint(0, 360)  # 隨機旋轉角度0-360度
    thickness = random.randint(3, 12)  # 邊框線條粗細範圍3-12像素
    
    # 計算正三角形的三個頂點（未旋轉）
    height = size * (3 ** 0.5) / 2  # 正三角形高度
    points = [
        (center_x, center_y - height ),  # 頂點（上）
        (center_x - size/1.5 , center_y + height / 3),  # 左下
        (center_x + size/1.5 , center_y + height / 3)   # 右下
    ]
    
    # 旋轉頂點
    rotated_points = []
    for x, y in points:
        new_x, new_y = rotate_point(x, y, angle, center_x, center_y)
        rotated_points.append((new_x, new_y))
    
    # 繪製中空三角形（僅邊框），顏色固定為黑色
    pygame.draw.polygon(surface, (0, 0, 0), rotated_points, thickness)
    
    # 儲存圖片
    filename = f"{OUTPUT_DIR}/triangle{index:1d}.png"
    pygame.image.save(surface, filename)
    return filename

async def main():
    # 生成10張圖片
    for i in range(1, NUM_IMAGES + 1):
        filename = generate_hollow_rotated_triangle_image(i)
        print(f"Generated: {filename}")
        await asyncio.sleep(0.01)  # 控制生成速度，模擬非阻塞行為
    
    print(f"Completed generating {NUM_IMAGES} images in {OUTPUT_DIR}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())