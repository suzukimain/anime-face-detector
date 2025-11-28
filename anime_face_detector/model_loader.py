"""
モデルのダウンロードとロード機能
"""
from __future__ import annotations

import hashlib
import pathlib
import warnings
from typing import Optional
from urllib.request import urlretrieve

import torch


# 公開されているアニメ顔検出モデルのURL
MODEL_URLS = {
    'yolov3': 'https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_yolov3.pth',
    'faster-rcnn': 'https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_faster-rcnn.pth',
    'hrnetv2': 'https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmpose_anime-face_hrnetv2.pth',
}

# モデルの保存先ディレクトリ
MODEL_DIR = pathlib.Path.home() / '.cache' / 'anime_face_detector'


def get_model_dir() -> pathlib.Path:
    """モデル保存ディレクトリを取得（なければ作成）"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def download_model(model_name: str, force_download: bool = False) -> pathlib.Path:
    """
    モデルをダウンロードする
    
    Args:
        model_name: モデル名 ('yolov3', 'faster-rcnn', 'hrnetv2')
        force_download: 既存ファイルを強制的に再ダウンロード
        
    Returns:
        ダウンロードしたモデルファイルのパス
    """
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_URLS.keys())}")
    
    model_dir = get_model_dir()
    model_path = model_dir / f'{model_name}.pth'
    
    if model_path.exists() and not force_download:
        print(f"Model already exists: {model_path}")
        return model_path
    
    url = MODEL_URLS[model_name]
    print(f"Downloading {model_name} model from {url}...")
    
    try:
        urlretrieve(url, model_path)
        print(f"Downloaded to: {model_path}")
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Failed to download model: {e}")
    
    return model_path


def load_model_weights(model_path: pathlib.Path) -> dict:
    """
    モデルの重みをロード
    
    Args:
        model_path: モデルファイルのパス
        
    Returns:
        モデルの state_dict
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # mmdetection/mmpose 形式のチェックポイントから state_dict を取得
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        return state_dict
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
