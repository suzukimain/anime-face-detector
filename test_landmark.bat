@echo off
REM アニメ顔ランドマーク検出テストバッチ
REM 使い方: test_landmark.bat <画像ファイルパス>

setlocal

REM ここでテストする画像を指定できます。空にするとコマンドライン引数を使います。
REM 例: set "input_img=assets\input.jpg"
set "input_img="

REM 入力画像の決定:
REM - バッチ内で input_img が設定されていればそれを使う
REM - そうでなければコマンドラインの最初の引数を入力画像とする
set "image_arg="
set "other_args="
if "%input_img%"=="" (
    if "%~1"=="" (
        echo 使い方: test_landmark.bat ^<画像ファイルパス^> [--detector ...] [--device ...]
        echo.
        echo 例: test_landmark.bat input.jpg
        echo 例: test_landmark.bat assets\input.jpg --detector yolov8s --device cpu
        echo.
        echo オプション:
        echo   --detector yolov8n^|yolov8s^|yolov8m^|yolov8l^|yolov8x
        echo   --device cpu^|cuda:0
        echo   --face-threshold 0.0-1.0
        echo   --landmark-threshold 0.0-1.0
        pause
        exit /b 1
    ) else (
        set "image_arg=%~1"
        shift
        set "other_args=%*"
    )
) else (
    set "image_arg=%input_img%"
    set "other_args=%*"
)

echo ===============================================
echo アニメ顔ランドマーク検出テスト
echo ===============================================
echo.

echo 入力画像: %image_arg%
if not "%other_args%"=="" (
    echo 追加引数: %other_args%
)

REM Pythonスクリプトを実行 (画像パスを先頭に渡す)
python test_landmark.py "%image_arg%" %other_args%

echo.
echo ===============================================
echo 処理完了
echo ===============================================
pause
