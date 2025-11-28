"""
テスト用: 指定した画像のランドマークを検出して保存するスクリプト
"""
import argparse
import pathlib
import sys

import cv2
import numpy as np

import anime_face_detector


def process_image(input_path: str, 
                  face_score_threshold: float = 0.5,
                  landmark_score_threshold: float = 0.3,
                  detector_name: str = 'yolov8n',
                  device: str = 'cpu'):
    """
    画像のランドマークを検出して保存する
    
    Args:
        input_path: 入力画像のパス
        face_score_threshold: 顔検出の閾値
        landmark_score_threshold: ランドマークの閾値
        detector_name: 使用するモデル名
        device: 使用するデバイス
    """
    input_path = pathlib.Path(input_path)
    
    if not input_path.exists():
        print(f"エラー: 画像ファイルが見つかりません: {input_path}")
        return False
    
    # 出力ファイル名を生成 (元のファイル名 + "_out" + 拡張子)
    output_path = input_path.parent / f"{input_path.stem}_out{input_path.suffix}"
    
    print(f"入力: {input_path}")
    print(f"出力: {output_path}")
    print(f"モデル: {detector_name}")
    print(f"デバイス: {device}")
    
    # 検出器を作成
    detector = anime_face_detector.create_detector(
        face_detector_name=detector_name,
        device=device
    )
    
    # 画像を読み込み
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"エラー: 画像を読み込めません: {input_path}")
        return False
    
    print(f"画像サイズ: {image.shape[1]}x{image.shape[0]}")
    
    # ランドマークを検出
    print("ランドマークを検出中...")
    preds = detector(image)
    
    if len(preds) == 0:
        print("警告: 顔が検出されませんでした")
    else:
        print(f"検出された顔の数: {len(preds)}")
    
    # 結果を描画
    res = image.copy()
    for i, pred in enumerate(preds):
        box = pred['bbox']
        box_coords, score = box[:4], box[4]
        
        if score < face_score_threshold:
            continue
        
        print(f"  顔 #{i+1}: スコア={score:.3f}")
        
        box_coords = np.round(box_coords).astype(int)
        
        # 線の太さを計算
        lt = max(2, int(3 * (box_coords[2:] - box_coords[:2]).max() / 256))
        
        # バウンディングボックスを描画
        cv2.rectangle(res, tuple(box_coords[:2]), tuple(box_coords[2:]), (0, 255, 0), lt)
        
        # ランドマークポイントを描画
        pred_pts = pred['keypoints']
        for j, (*pt, lm_score) in enumerate(pred_pts):
            if lm_score < landmark_score_threshold:
                color = (0, 255, 255)  # 黄色 (低スコア)
            else:
                color = (0, 0, 255)  # 赤 (高スコア)
            pt = np.round(pt).astype(int)
            cv2.circle(res, tuple(pt), lt, color, cv2.FILLED)
    
    # 画像を保存
    cv2.imwrite(str(output_path), res)
    print(f"保存完了: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='画像のランドマークを検出して保存するテストスクリプト'
    )
    parser.add_argument('input', 
                       type=str, 
                       help='入力画像のパス')
    parser.add_argument('--detector',
                       type=str,
                       default='yolov8n',
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='使用するYOLOモデル (デフォルト: yolov8n)')
    parser.add_argument('--device',
                       type=str,
                       default='cpu',
                       help='使用するデバイス (デフォルト: cpu)')
    parser.add_argument('--face-threshold',
                       type=float,
                       default=0.5,
                       help='顔検出の閾値 (デフォルト: 0.5)')
    parser.add_argument('--landmark-threshold',
                       type=float,
                       default=0.3,
                       help='ランドマーク検出の閾値 (デフォルト: 0.3)')
    
    args = parser.parse_args()
    
    success = process_image(
        input_path=args.input,
        face_score_threshold=args.face_threshold,
        landmark_score_threshold=args.landmark_threshold,
        detector_name=args.detector,
        device=args.device
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
