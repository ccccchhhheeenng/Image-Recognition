# Image Recognition

本專案使用 **Python** 進行影像辨識，並包含相關模型與權重檔案。

## 📌 功能
- 識別不同形狀，例如：
  - 圓形（circle）
  - 十字（cross）
  - 立方體（cube）
  - 三角形（triangle）
- 支援 **灰階處理**（grayscale）
- 使用 **PyTorch** 進行模型運算
- 內含模型權重 (`bias_hidden.pt`, `bias_hidden.txt`)

## ⚡ 安裝
請確保您的環境中已安裝 Python，然後執行以下命令以安裝所需套件：
```bash
pip install -r requirements.txt
```

## 🚀 使用方法
1. 將您的影像放入 `temp/` 目錄中。
2. 執行 Python 腳本來進行影像辨識：
```bash
python recognize.py --image temp/sample.png
```
3. 結果將顯示在終端機，或存入輸出檔案。

## 📝 檔案結構
```
Image-Recognition/
│── temp/               # 存放輸入影像
│── bias_hidden.pt      # 模型權重
│── bias_hidden.txt     # 模型權重 (文字格式)
│── recognize.py        # 影像辨識程式
│── requirements.txt    # 依賴套件
│── LICENSE            # 授權條款 (MIT)
│── README.md          # 專案說明文件
```

## 🏆 貢獻
如果你有興趣改進本專案，歡迎提交 Pull Request！

## 📜 授權
本專案遵循 [MIT License](LICENSE)，可自由使用與修改。
