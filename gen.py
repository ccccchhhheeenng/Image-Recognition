import pygame
import random
import asyncio
import platform
import os

# 初始化Pygame
pygame.init()

# 設定參數
WIDTH, HEIGHT = 500, 500
OUTPUT_DIR = "cross/image"
NUM_IMAGES = 1000

# 確保輸出目錄存在（模擬檔案系統，Pyodide環境下實際由瀏覽器處理）
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except:
    pass

def generate_hollow_cross_image(index):
    # 創建表面
    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill((255, 255, 255))  # 白色背景
    
    # 隨機十字架參數
    arm_length = random.randint(100, 250)  # 手臂長度範圍40-150像素
    arm_width = random.randint(2, 15)  # 手臂粗細範圍2-10像素
    center_x = random.randint(arm_length, WIDTH - arm_length)  # 確保十字架不超出邊界
    center_y = random.randint(arm_length, HEIGHT - arm_length)
    
    color = (0,0,0)  # 隨機顏色
    
    # 繪製垂直線
    pygame.draw.line(surface, color, (center_x, center_y - arm_length // 2), (center_x, center_y + arm_length // 2), arm_width)
    # 繪製水平線
    pygame.draw.line(surface, color, (center_x - arm_length // 2, center_y), (center_x + arm_length // 2, center_y), arm_width)
    
    # 儲存圖片
    filename = f"{OUTPUT_DIR}/cross{index:1d}.png"
    pygame.image.save(surface, filename)
    return filename

async def main():
    # 生成10張圖片
    for i in range(1, NUM_IMAGES + 1):
        filename = generate_hollow_cross_image(i)
        print(f"Generated: {filename}")
        await asyncio.sleep(0.01)  # 控制生成速度，模擬非阻塞行為
    
    print(f"Completed generating {NUM_IMAGES} images in {OUTPUT_DIR}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())