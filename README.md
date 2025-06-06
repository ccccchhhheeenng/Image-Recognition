# Image Recognition

本專案使用 **Python** 進行影像辨識，並包含相關模型與權重檔案。

## 📌 功能
- **影像繪製與辨識**：使用 `draw.py` 產生圖像後進行分析
- 識別不同形狀，例如：
  - 圓形（circle）
  - 十字（cross）
  - 立方體（cube）
  - 三角形（triangle）
- 支援 **灰階處理**（grayscale）
- 使用 **PyTorch** 進行模型運算
- 內含模型權重,可直接運行辨識

## ⚡ 安裝
請確保您的環境中已安裝 Python和pip環境，如果需要訓練且設備支援cuda，建議先執行以下命令以安裝所需套件：
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 🚀 使用方法
1. 執行 `draw.py` 來繪製影像並辨識：
```bash
python draw.py 
```
2. 結果將顯示在GUI，並列出個形狀的機率。

3. 可透過自行運行training.py進行模型訓練,預設是進行20萬次、隱藏層512層、訓練率0.0015，可以依照需求自行變更
```bash
python training.py 
```

## 📝 檔案結構
```
Image-Recognition/
│── temp/               # 存放輸入影像
│── training            #訓練
│── draw.py             # 影像繪製程式
│── LICENSE            # 授權條款 (MIT)
│── README.md          # 專案說明文件
```

## 🏆 貢獻
如果你有興趣改進本專案，歡迎提交 Pull Request！

## 📜 授權
本專案遵循 [MIT License](LICENSE)，可自由使用與修改。
