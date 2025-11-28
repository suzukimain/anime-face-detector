@echo off
REM アニメ顔ランドマーク検出テストバッチ
REM 使い方: test_landmark.bat <画像ファイルパス>

setlocal

if "%~1"=="" (
    echo 使い方: test_landmark.bat ^<画像ファイルパス^>
    echo.
    echo 例: test_landmark.bat input.jpg
    echo 例: test_landmark.bat assets\input.jpg
    echo 例: test_landmark.bat "C:\path\to\image.png"
    echo.
    echo オプション:
    echo   --detector yolov8n^|yolov8s^|yolov8m^|yolov8l^|yolov8x
    echo   --device cpu^|cuda:0
    echo   --face-threshold 0.0-1.0
    echo   --landmark-threshold 0.0-1.0
    pause
    exit /b 1
)

echo ===============================================
echo アニメ顔ランドマーク検出テスト
echo ===============================================
echo.

REM Pythonスクリプトを実行
python test_landmark.py %*

echo.
echo ===============================================
echo 処理完了
echo ===============================================
pause
