"""
MMDetection/MMPose ベースのアニメ顔ランドマーク検出器
オリジナルの学習済みモデルを使用
"""
from __future__ import annotations

import pathlib
import warnings
from typing import Optional, Union

import cv2
import numpy as np
import torch

from .model_loader import download_model


class MMDetLandmarkDetector:
    """
    MMDetection と MMPose を使用したアニメ顔ランドマーク検出器
    オリジナルの学習済みモデル (yolov3, faster-rcnn, hrnetv2) を使用
    """
    
    def __init__(
            self,
            face_detector: str = 'yolov3',
            landmark_model: str = 'hrnetv2',
            device: str = 'cuda:0',
            box_scale_factor: float = 1.1,
            face_score_threshold: float = 0.5):
        """
        Initialize the landmark detector.
        
        Args:
            face_detector: Face detection model ('yolov3' or 'faster-rcnn')
            landmark_model: Landmark detection model ('hrnetv2')
            device: Device to run inference on ('cuda:0', 'cpu', etc.)
            box_scale_factor: Factor to scale bounding boxes
            face_score_threshold: Minimum confidence for face detection
        """
        self.device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
        self.box_scale_factor = box_scale_factor
        self.face_score_threshold = face_score_threshold
        
        # Check if mmdet and mmpose are available
        try:
            import mmdet
            import mmpose
            import mmcv
            from mmdet.apis import init_detector, inference_detector
            from mmpose.apis import init_model as init_pose_model, inference_topdown
            self.mmdet_available = True
        except ImportError as e:
            self.mmdet_available = False
            raise ImportError(
                "MMDetection and MMPose are required for using original anime face models.\n"
                "Please install them:\n"
                "  pip install openmim\n"
                "  mim install mmcv>=2.0.0\n"
                "  pip install mmdet>=3.0.0 mmpose>=1.0.0\n"
                f"Error: {e}"
            )
        
        # Download models if needed
        face_model_path = download_model(face_detector)
        landmark_model_path = download_model(landmark_model)
        
        # Get config paths
        config_dir = pathlib.Path(__file__).parent / 'configs'
        face_config = config_dir / 'mmdet' / f'{face_detector}.py'
        landmark_config = config_dir / 'mmpose' / f'{landmark_model}.py'
        
        if not face_config.exists():
            raise FileNotFoundError(f"Config file not found: {face_config}")
        if not landmark_config.exists():
            raise FileNotFoundError(f"Config file not found: {landmark_config}")
        
        # Initialize models
        print(f"Loading face detector: {face_detector}")
        self.face_detector = init_detector(
            str(face_config), 
            str(face_model_path), 
            device=self.device
        )
        
        print(f"Loading landmark model: {landmark_model}")
        self.landmark_detector = init_pose_model(
            str(landmark_config),
            str(landmark_model_path),
            device=self.device
        )
        
        self.inference_detector = inference_detector
        self.inference_topdown = inference_topdown
        self.num_landmarks = 28

    def _detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Detect faces using MMDetection model.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of bounding boxes [x0, y0, x1, y1, score]
        """
        result = self.inference_detector(self.face_detector, image)
        
        # Extract bounding boxes
        boxes = []
        if hasattr(result, 'pred_instances'):
            # MMDet 3.x format
            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            
            for bbox, score in zip(bboxes, scores):
                if score >= self.face_score_threshold:
                    box = np.array([bbox[0], bbox[1], bbox[2], bbox[3], score], dtype=np.float32)
                    boxes.append(box)
        else:
            # MMDet 2.x format
            if isinstance(result, tuple):
                bbox_result = result[0]
            else:
                bbox_result = result
            
            if isinstance(bbox_result, list):
                bbox_result = bbox_result[0]  # First class
            
            for bbox in bbox_result:
                if bbox[4] >= self.face_score_threshold:
                    box = np.array(bbox, dtype=np.float32)
                    boxes.append(box)
        
        # Scale boxes
        boxes = self._update_pred_box(boxes)
        return boxes

    def _update_pred_box(self, pred_boxes: list[np.ndarray]) -> list[np.ndarray]:
        """Scale bounding boxes by box_scale_factor."""
        boxes = []
        for pred_box in pred_boxes:
            box = pred_box[:4].copy()
            size = box[2:] - box[:2] + 1
            new_size = size * self.box_scale_factor
            center = (box[:2] + box[2:]) / 2
            tl = center - new_size / 2
            br = tl + new_size
            new_box = pred_box.copy()
            new_box[:4] = np.concatenate([tl, br])
            boxes.append(new_box)
        return boxes

    def _detect_landmarks(
            self, image: np.ndarray,
            boxes: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        """
        Detect landmarks for given face bounding boxes using MMPose.
        """
        preds = []
        
        for box in boxes:
            bbox_xyxy = box[:4].reshape(1, 4)
            
            # Inference with MMPose
            pose_results = self.inference_topdown(
                self.landmark_detector,
                image,
                bbox_xyxy,
                bbox_format='xyxy'
            )
            
            if len(pose_results) > 0:
                pose_result = pose_results[0]
                
                # Extract keypoints
                if hasattr(pose_result, 'pred_instances'):
                    # MMPose 1.x format
                    keypoints = pose_result.pred_instances.keypoints[0]  # [28, 2]
                    scores = pose_result.pred_instances.keypoint_scores[0]  # [28]
                    
                    # Combine to [28, 3] format
                    landmarks = np.concatenate([keypoints, scores[:, None]], axis=1)
                else:
                    # Older format
                    landmarks = pose_result.get('keypoints', np.zeros((28, 3)))
                
                pred = {
                    'bbox': box,
                    'keypoints': landmarks.astype(np.float32)
                }
                preds.append(pred)
        
        return preds

    @staticmethod
    def _load_image(
            image_or_path: Union[np.ndarray, str, pathlib.Path]) -> np.ndarray:
        """Load image from path or return existing numpy array."""
        if isinstance(image_or_path, np.ndarray):
            image = image_or_path
        elif isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        elif isinstance(image_or_path, pathlib.Path):
            image = cv2.imread(str(image_or_path))
        else:
            raise ValueError("Invalid image input type")
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_or_path}")
        
        return image

    def __call__(
        self,
        image_or_path: Union[np.ndarray, str, pathlib.Path],
        boxes: Optional[list[np.ndarray]] = None
    ) -> list[dict[str, np.ndarray]]:
        """
        Detect face landmarks.

        Args:
            image_or_path: An image with BGR channel order or an image path.
            boxes: Optional list of bounding boxes [x0, y0, x1, y1, [score]].
                   If None, faces will be detected automatically.

        Returns:
            List of detection results. Each result contains:
                - 'bbox': [x0, y0, x1, y1, score]
                - 'keypoints': array of shape (28, 3) with [x, y, confidence]
        """
        image = self._load_image(image_or_path)
        
        if boxes is None:
            boxes = self._detect_faces(image)
            
            if len(boxes) == 0:
                warnings.warn('No faces detected.')
                return []
        
        return self._detect_landmarks(image, boxes)
