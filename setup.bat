@echo off
REM setup.bat
REM: 仮想環境を作成・初期化し、オプションでテスト実行も行うヘルパー

setlocal EnableDelayedExpansion

REM 文字化け対策: コンソールを UTF-8 に設定し、Python 側でも UTF-8 を有効化
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM 仮想環境のディレクトリ
set "VENV_DIR=venv"

echo ===============================================
echo 仮想環境セットアップスクリプト
echo ===============================================

REM venv が存在しない場合は作成して依存関係をインストール
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo 仮想環境 "%VENV_DIR%" を作成します...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo エラー: 仮想環境の作成に失敗しました。システムの Python が必要です。
        pause
        exit /b 1
    )

    echo pip をアップグレードしています...
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip

    if exist "requirements.txt" (
        echo requirements.txt からパッケージをインストールします...
        "%VENV_DIR%\Scripts\python.exe" -m pip install -r "requirements.txt"
    ) else (
        echo requirements.txt が見つかりません。必要なパッケージを手動でインストールしてください。
    )
) else (
    echo 仮想環境 "%VENV_DIR%" は既に存在します。
)

echo.
echo 仮想環境の準備ができました。
echo 使用方法:
echo   .\setup.bat --run-test "<画像パス>" [--detector yolov8n --device cpu]
echo   あるいは:
echo   pushd %%~dp0
echo   .\%%VENV_DIR%%\Scripts\activate.bat
echo   .\%%VENV_DIR%%\Scripts\python.exe test_landmark.py "<画像パス>"
echo.

REM オプション: --run-test を指定すると、続く引数を test_landmark.py に渡して実行
if "%~1"=="--run-test" (
    shift
    if "%~1"=="" (
        echo エラー: --run-test を指定しましたが画像パスがありません。
        pause
        exit /b 1
    )
    echo テスト実行: %*
    "%VENV_DIR%\Scripts\python.exe" "%~dp0test_landmark.py" %*
    echo テストが終了しました。
)

pause
