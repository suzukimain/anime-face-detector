@echo off
REM setup_mmdet.bat
REM: MMDetection と MMPose をインストールしてオリジナルモデルを使用可能にする

setlocal EnableDelayedExpansion

REM 文字化け対策
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "VENV_DIR=venv"

echo ===============================================
echo MMDetection/MMPose セットアップ
echo ===============================================
echo.
echo このスクリプトは以下をインストールします:
echo   - openmim
echo   - mmcv ^>= 2.0.0
echo   - mmdet ^>= 3.0.0
echo   - mmpose ^>= 1.0.0
echo.
echo 注意: Windows では Visual Studio C++ Build Tools が必要です
echo       インストールされていない場合は先にインストールしてください
echo       https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo.

pause

REM 仮想環境の確認
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo エラー: 仮想環境が見つかりません。先に setup.bat を実行してください。
    pause
    exit /b 1
)

echo 仮想環境を使用します: %VENV_DIR%
echo.

echo openmim をインストール中...
"%VENV_DIR%\Scripts\python.exe" -m pip install openmim
if errorlevel 1 (
    echo エラー: openmim のインストールに失敗しました
    pause
    exit /b 1
)

echo mmcv をインストール中...
"%VENV_DIR%\Scripts\python.exe" -m mim install "mmcv>=2.0.0"
if errorlevel 1 (
    echo エラー: mmcv のインストールに失敗しました
    pause
    exit /b 1
)

echo mmdet をインストール中...
"%VENV_DIR%\Scripts\python.exe" -m pip install "mmdet>=3.0.0"
if errorlevel 1 (
    echo エラー: mmdet のインストールに失敗しました
    pause
    exit /b 1
)

echo mmpose をインストール中...
"%VENV_DIR%\Scripts\python.exe" -m pip install "mmpose>=1.0.0"
if errorlevel 1 (
    echo エラー: mmpose のインストールに失敗しました
    pause
    exit /b 1
)

echo.
echo ===============================================
echo インストール完了
echo ===============================================
echo.
echo これでオリジナルのアニメ顔検出モデルを使用できます。
echo.
echo 使用例:
echo   .\test_landmark.bat test.jpg --detector yolov3 --use-mmdet
echo   .\test_landmark.bat test.jpg --detector faster-rcnn --use-mmdet
echo.

pause
