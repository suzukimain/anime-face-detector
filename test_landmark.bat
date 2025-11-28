@echo off
REM アニメ顔ランドマーク検出テストバッチ
REM 使い方: test_landmark.bat <画像ファイルパス>

setlocal EnableDelayedExpansion

REM 文字化け対策: コンソールを UTF-8 に設定し、Python 側でも UTF-8 を有効化
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM 仮想環境のディレクトリ (必要に応じて変更してください)
set "VENV_DIR=venv"

REM ここでテストする画像を指定できます。空にするとコマンドライン引数を使います。
REM 例: set "input_img=assets\input.jpg"
set "input_img=F:\github\anime-face-detector\assets\Hoshino.png"

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

REM 仮想環境が無ければ作成して依存関係をインストール
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo 仮想環境 "%VENV_DIR%" が見つかりません。作成します...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo 仮想環境の作成に失敗しました。システムの Python が必要です。
        pause
        exit /b 1
    )
    echo pip をアップグレードします...
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip >nul
    if exist "requirements.txt" (
        echo requirements.txt からパッケージをインストールします...
        "%VENV_DIR%\Scripts\python.exe" -m pip install -r "requirements.txt"
    ) else (
        echo requirements.txt が見つかりません。必要なパッケージを手動でインストールしてください。
    )
)

REM 仮想環境の Python を使ってスクリプト実行 (画像パスを先頭に渡す)
echo 仮想環境を使用してスクリプトを実行します: %VENV_DIR%\Scripts\python.exe
"%VENV_DIR%\Scripts\python.exe" "%~dp0test_landmark.py" "%image_arg%" %other_args%

echo.
echo ===============================================
echo 処理完了
echo ===============================================
pause
