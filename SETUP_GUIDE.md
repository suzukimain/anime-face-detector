# アニメ顔ランドマーク検出 - セットアップガイド

このリポジトリは、アニメ顔の検出とランドマーク（顔の特徴点）を検出するツールです。

## 2つの検出方法

### 方法1: Ultralytics YOLO (簡単・Windows互換)
- **特徴**: インストールが簡単、依存関係が少ない
- **精度**: 人間の顔用モデルをベースにしたキーポイント検出
- **推奨**: 手軽に試したい場合

### 方法2: MMDetection + MMPose (高精度・オリジナルモデル)
- **特徴**: アニメ顔専用に学習されたモデル
- **精度**: アニメ顔に特化した28個のランドマーク検出
- **必要**: Visual Studio C++ Build Tools、追加インストール
- **推奨**: 最高の精度が必要な場合

## セットアップ手順

### 1. 基本セットアップ (必須)

```powershell
# 仮想環境を作成して基本パッケージをインストール
.\setup.bat
```

このスクリプトは以下を実行します:
- Python 仮想環境 (`venv`) を作成
- `requirements.txt` から基本パッケージをインストール
- ultralytics, numpy, opencv-python, torch など

### 2. オリジナルモデル使用の追加セットアップ (オプション)

オリジナルのアニメ顔検出モデルを使用する場合:

#### Windows の場合
1. **Visual Studio C++ Build Tools をインストール**
   - https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - "Desktop development with C++" を選択してインストール

2. **MMDetection/MMPose をインストール**
   ```powershell
   .\setup_mmdet.bat
   ```

#### Linux/Mac の場合
```bash
source venv/bin/activate
pip install openmim
mim install "mmcv>=2.0.0"
pip install "mmdet>=3.0.0" "mmpose>=1.0.0"
```

## 使い方

### 基本的な使い方 (Ultralytics YOLO)

```powershell
# 画像パスを直接指定
.\test_landmark.bat test_image.jpg

# モデルとデバイスを指定
.\test_landmark.bat test_image.jpg --detector yolov8s --device cpu

# より高精度なモデルを使用
.\test_landmark.bat test_image.jpg --detector yolov8m --device cuda:0
```

### オリジナルモデルを使用 (MMDetection)

```powershell
# YOLOv3 ベース (軽量)
.\test_landmark.bat test_image.jpg --detector yolov3 --use-mmdet

# Faster R-CNN ベース (高精度)
.\test_landmark.bat test_image.jpg --detector faster-rcnn --use-mmdet --device cuda:0
```

### バッチファイル内で画像を指定

`test_landmark.bat` を編集して、`input_img` 変数に画像パスを設定:

```bat
set "input_img=assets\input.jpg"
```

その後、引数なしで実行:
```powershell
.\test_landmark.bat
```

### セットアップ時にテスト実行

```powershell
.\setup.bat --run-test test_image.jpg --detector yolov8n --device cpu
```

## オプション

- `--detector`: 使用するモデル
  - Ultralytics: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
  - MMDetection: `yolov3`, `faster-rcnn`
- `--device`: 実行デバイス (`cpu`, `cuda:0`, `cuda:1`, など)
- `--face-threshold`: 顔検出の信頼度閾値 (0.0-1.0, デフォルト: 0.5)
- `--landmark-threshold`: ランドマーク表示の信頼度閾値 (0.0-1.0, デフォルト: 0.3)
- `--use-mmdet`: オリジナルのMMDetectionベースモデルを使用

## 出力

処理された画像は元のファイル名に `_out` を付けて保存されます:
- 入力: `test_image.jpg`
- 出力: `test_image_out.jpg`

出力画像には以下が描画されます:
- **緑色の矩形**: 検出された顔のバウンディングボックス
- **赤色の点**: 高信頼度のランドマークポイント
- **黄色の点**: 低信頼度のランドマークポイント

## トラブルシューティング

### MMDetection のインストールに失敗する

**Windows**: Visual Studio C++ Build Tools が必要です
1. https://visualstudio.microsoft.com/visual-cpp-build-tools/ からインストール
2. "Desktop development with C++" を選択
3. 再度 `.\setup_mmdet.bat` を実行

### CUDA が使えない

```powershell
# CPU で実行
.\test_landmark.bat test.jpg --device cpu
```

### モデルのダウンロードに失敗する

初回実行時、モデルファイルが自動ダウンロードされます:
- Ultralytics: `~/.cache/torch/hub/ultralytics/`
- オリジナル: `~/.cache/anime_face_detector/`

ネットワーク接続を確認してください。

## モデルサイズと速度

### Ultralytics YOLO
- `yolov8n-pose`: 約6MB (最速、軽量)
- `yolov8s-pose`: 約11MB
- `yolov8m-pose`: 約26MB (バランス)
- `yolov8l-pose`: 約44MB
- `yolov8x-pose`: 約69MB (最高精度、低速)

### オリジナルモデル
- `yolov3`: 約248MB (軽量、アニメ顔特化)
- `faster-rcnn`: 約167MB (高精度、アニメ顔特化)
- `hrnetv2`: 約78MB (ランドマーク検出用)

## ライセンス

このプロジェクトは元の anime-face-detector に基づいています。
詳細は LICENSE ファイルを参照してください。
