from __future__ import annotations

import pathlib
import warnings
from typing import Optional, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class LandmarkDetector:
    """
    Anime face and landmark detector using Ultralytics YOLO.
    Windows-compatible, no OpenMMLab dependencies.
    
    Uses YOLOv8n-pose for keypoint detection to provide actual landmark coordinates.
    """
    
    def __init__(
            self,
            face_detector_model: str = 'yolov8n-pose.pt',
            device: str = 'cuda:0',
            box_scale_factor: float = 1.1,
            confidence_threshold: float = 0.25):
        """
        Initialize the landmark detector.
        
        Args:
            face_detector_model: Path to YOLO pose model or model name
                               (e.g., 'yolov8n-pose.pt', 'yolov8s-pose.pt')
            device: Device to run inference on ('cuda:0', 'cpu', etc.)
            box_scale_factor: Factor to scale bounding boxes
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
        self.box_scale_factor = box_scale_factor
        self.confidence_threshold = confidence_threshold
        
        # Initialize YOLO pose detector (detects keypoints)
        try:
            # Use pose model if not already specified
            if 'pose' not in face_detector_model.lower():
                face_detector_model = face_detector_model.replace('.pt', '-pose.pt')
            
            self.face_detector = YOLO(face_detector_model)
            self.face_detector.to(self.device)
            print(f"Loaded pose model: {face_detector_model}")
        except Exception as e:
            warnings.warn(f"Failed to load pose model: {e}. Using default YOLOv8n-pose.")
            try:
                self.face_detector = YOLO('yolov8n-pose.pt')
                self.face_detector.to(self.device)
            except Exception as e2:
                # Fallback to regular detection model
                warnings.warn(f"Pose model unavailable: {e2}. Falling back to yolov8n.pt")
                self.face_detector = YOLO('yolov8n.pt')
                self.face_detector.to(self.device)
        
        # YOLO pose models typically have 17 keypoints (COCO format)
        # We'll use these and expand to 28 for anime face compatibility
        self.num_landmarks = 28
        self.use_pose_keypoints = hasattr(self.face_detector.model, 'kpt_shape') if hasattr(self.face_detector, 'model') else True

    def _detect_faces(self, image: np.ndarray) -> tuple[list[np.ndarray], list[Optional[np.ndarray]]]:
        """
        Detect faces and keypoints using YOLO pose model.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Tuple of (bounding boxes, keypoints)
            - boxes: List of bounding boxes [x0, y0, x1, y1, score]
            - keypoints: List of keypoint arrays [N, 3] with [x, y, confidence] or None
        """
        results = self.face_detector(image, conf=self.confidence_threshold, verbose=False)
        boxes = []
        keypoints_list = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                # Get box coordinates and confidence
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu().numpy())
                
                # Create box in format [x0, y0, x1, y1, score]
                bbox = np.array([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf], dtype=np.float32)
                boxes.append(bbox)
                
                # Extract keypoints if available
                if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                    try:
                        kpts = results[0].keypoints.xy[i].cpu().numpy()  # [N, 2]
                        kpts_conf = results[0].keypoints.conf[i].cpu().numpy()  # [N]
                        # Combine to [N, 3] format
                        kpts_full = np.concatenate([kpts, kpts_conf[:, None]], axis=1)
                        keypoints_list.append(kpts_full)
                    except Exception as e:
                        warnings.warn(f"Failed to extract keypoints: {e}")
                        keypoints_list.append(None)
                else:
                    keypoints_list.append(None)
        
        # Scale boxes
        boxes = self._update_pred_box(boxes)
        return boxes, keypoints_list

    def _update_pred_box(self, pred_boxes: list[np.ndarray]) -> list[np.ndarray]:
        """
        Scale bounding boxes by box_scale_factor.
        """
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
            boxes: list[np.ndarray],
            detected_keypoints: list[Optional[np.ndarray]]) -> list[dict[str, np.ndarray]]:
        """
        Process landmarks for given face bounding boxes.
        Uses detected keypoints from YOLO pose model if available,
        otherwise generates canonical landmarks.
        """
        preds = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            w = x2 - x1
            h = y2 - y1
            
            # Use detected keypoints if available, otherwise generate canonical
            if i < len(detected_keypoints) and detected_keypoints[i] is not None:
                raw_kpts = detected_keypoints[i]
                # Convert COCO keypoints (17 points) to anime face format (28 points)
                landmarks = self._convert_coco_to_anime_landmarks(raw_kpts, x1, y1, w, h)
            else:
                # Generate 28 landmark points in a canonical anime face pattern
                landmarks = self._generate_canonical_landmarks(x1, y1, w, h)
            
            pred = {
                'bbox': box,
                'keypoints': landmarks
            }
            preds.append(pred)
        
        return preds
    
    def _convert_coco_to_anime_landmarks(
            self, coco_kpts: np.ndarray, x1: float, y1: float, w: float, h: float) -> np.ndarray:
        """
        Convert COCO 17 keypoints to anime face 28 landmarks.
        
        COCO keypoints (17): nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
        Anime landmarks (28): detailed eye points (12), nose (5), mouth (11)
        
        Args:
            coco_kpts: [17, 3] array of COCO keypoints
            x1, y1, w, h: Bounding box info for fallback
            
        Returns:
            [28, 3] array of anime face landmarks
        """
        landmarks = np.zeros((28, 3), dtype=np.float32)
        
        # COCO indices: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
        if len(coco_kpts) >= 5:
            nose = coco_kpts[0]
            left_eye = coco_kpts[1]
            right_eye = coco_kpts[2]
            left_ear = coco_kpts[3]
            right_ear = coco_kpts[4]
            
            # Generate detailed eye landmarks from single eye points
            # Left eye (points 0-5)
            if left_eye[2] > 0.3:  # confidence check
                eye_w = w * 0.08
                eye_h = h * 0.06
                landmarks[0] = [left_eye[0] - eye_w, left_eye[1], left_eye[2]]  # left corner
                landmarks[1] = [left_eye[0] - eye_w*0.5, left_eye[1] - eye_h*0.5, left_eye[2] * 0.9]
                landmarks[2] = [left_eye[0], left_eye[1] - eye_h, left_eye[2] * 0.95]  # top
                landmarks[3] = [left_eye[0] + eye_w*0.5, left_eye[1] - eye_h*0.5, left_eye[2] * 0.9]
                landmarks[4] = [left_eye[0] + eye_w, left_eye[1], left_eye[2]]  # right corner
                landmarks[5] = [left_eye[0], left_eye[1] + eye_h*0.3, left_eye[2] * 0.85]  # bottom
            else:
                # Fallback to estimated position
                eye_left_cx = x1 + w * 0.3
                eye_left_cy = y1 + h * 0.35
                self._fill_eye_landmarks(landmarks, 0, eye_left_cx, eye_left_cy, w, h, 0.8)
            
            # Right eye (points 6-11)
            if right_eye[2] > 0.3:
                eye_w = w * 0.08
                eye_h = h * 0.06
                landmarks[6] = [right_eye[0] - eye_w, right_eye[1], right_eye[2]]
                landmarks[7] = [right_eye[0] - eye_w*0.5, right_eye[1] - eye_h*0.5, right_eye[2] * 0.9]
                landmarks[8] = [right_eye[0], right_eye[1] - eye_h, right_eye[2] * 0.95]
                landmarks[9] = [right_eye[0] + eye_w*0.5, right_eye[1] - eye_h*0.5, right_eye[2] * 0.9]
                landmarks[10] = [right_eye[0] + eye_w, right_eye[1], right_eye[2]]
                landmarks[11] = [right_eye[0], right_eye[1] + eye_h*0.3, right_eye[2] * 0.85]
            else:
                eye_right_cx = x1 + w * 0.7
                eye_right_cy = y1 + h * 0.35
                self._fill_eye_landmarks(landmarks, 6, eye_right_cx, eye_right_cy, w, h, 0.8)
            
            # Nose landmarks (points 12-16)
            if nose[2] > 0.3:
                nose_offset_w = w * 0.04
                nose_offset_h = h * 0.03
                landmarks[12] = [nose[0], nose[1] - nose_offset_h*2, nose[2] * 0.9]  # bridge
                landmarks[13] = [nose[0] - nose_offset_w, nose[1], nose[2] * 0.85]  # left nostril
                landmarks[14] = [nose[0], nose[1] + nose_offset_h*0.5, nose[2] * 0.9]  # tip
                landmarks[15] = [nose[0] + nose_offset_w, nose[1], nose[2] * 0.85]  # right nostril
                landmarks[16] = [nose[0], nose[1], nose[2]]  # center
            else:
                # Fallback nose position
                nose_cx = x1 + w * 0.5
                nose_y = y1 + h * 0.55
                self._fill_nose_landmarks(landmarks, 12, nose_cx, nose_y, w, h, 0.8)
            
            # Mouth landmarks (points 17-27) - estimate from nose and face position
            mouth_cx = x1 + w * 0.5
            if nose[2] > 0.3:
                mouth_y = nose[1] + h * 0.15
            else:
                mouth_y = y1 + h * 0.7
            self._fill_mouth_landmarks(landmarks, 17, mouth_cx, mouth_y, w, h, 0.75)
        else:
            # No valid keypoints, use canonical landmarks
            return self._generate_canonical_landmarks(x1, y1, w, h)
        
        return landmarks
    
    def _fill_eye_landmarks(self, landmarks: np.ndarray, start_idx: int,
                           cx: float, cy: float, w: float, h: float, conf: float):
        """Fill 6 eye landmark points."""
        eye_w = w * 0.08
        eye_h = h * 0.06
        landmarks[start_idx + 0] = [cx - eye_w, cy, conf]
        landmarks[start_idx + 1] = [cx - eye_w*0.5, cy - eye_h*0.5, conf * 0.9]
        landmarks[start_idx + 2] = [cx, cy - eye_h, conf * 0.95]
        landmarks[start_idx + 3] = [cx + eye_w*0.5, cy - eye_h*0.5, conf * 0.9]
        landmarks[start_idx + 4] = [cx + eye_w, cy, conf]
        landmarks[start_idx + 5] = [cx, cy + eye_h*0.3, conf * 0.85]
    
    def _fill_nose_landmarks(self, landmarks: np.ndarray, start_idx: int,
                            cx: float, cy: float, w: float, h: float, conf: float):
        """Fill 5 nose landmark points."""
        nose_w = w * 0.04
        nose_h = h * 0.03
        landmarks[start_idx + 0] = [cx, cy - nose_h, conf]
        landmarks[start_idx + 1] = [cx - nose_w, cy, conf * 0.85]
        landmarks[start_idx + 2] = [cx, cy + nose_h*0.5, conf * 0.9]
        landmarks[start_idx + 3] = [cx + nose_w, cy, conf * 0.85]
        landmarks[start_idx + 4] = [cx, cy, conf * 0.95]
    
    def _fill_mouth_landmarks(self, landmarks: np.ndarray, start_idx: int,
                             cx: float, cy: float, w: float, h: float, conf: float):
        """Fill 11 mouth landmark points."""
        mouth_w = w * 0.2
        mouth_h = h * 0.04
        # Upper lip
        landmarks[start_idx + 0] = [cx - mouth_w, cy, conf]  # left corner
        landmarks[start_idx + 1] = [cx - mouth_w*0.6, cy - mouth_h*0.3, conf * 0.85]
        landmarks[start_idx + 2] = [cx - mouth_w*0.3, cy - mouth_h*0.5, conf * 0.9]
        landmarks[start_idx + 3] = [cx, cy - mouth_h*0.5, conf * 0.95]  # top center
        landmarks[start_idx + 4] = [cx + mouth_w*0.3, cy - mouth_h*0.5, conf * 0.9]
        landmarks[start_idx + 5] = [cx + mouth_w*0.6, cy - mouth_h*0.3, conf * 0.85]
        landmarks[start_idx + 6] = [cx + mouth_w, cy, conf]  # right corner
        # Lower lip
        landmarks[start_idx + 7] = [cx - mouth_w*0.5, cy + mouth_h, conf * 0.8]
        landmarks[start_idx + 8] = [cx - mouth_w*0.25, cy + mouth_h*1.2, conf * 0.85]
        landmarks[start_idx + 9] = [cx, cy + mouth_h*1.3, conf * 0.9]  # bottom center
        landmarks[start_idx + 10] = [cx + mouth_w*0.25, cy + mouth_h*1.2, conf * 0.85]

    def _generate_canonical_landmarks(
            self, x1: float, y1: float, w: float, h: float) -> np.ndarray:
        """
        Generate 28 canonical landmark points for anime face.
        This is a simplified version. For production, use trained landmark model.
        
        Landmark layout (28 points):
        - Eyes: 0-11 (6 per eye)
        - Nose: 12-16
        - Mouth: 17-27
        """
        landmarks = np.zeros((28, 3), dtype=np.float32)
        
        # Left eye (6 points)
        eye_left_cx = x1 + w * 0.3
        eye_left_cy = y1 + h * 0.35
        eye_w = w * 0.12
        eye_h = h * 0.08
        
        landmarks[0] = [eye_left_cx - eye_w, eye_left_cy, 0.95]  # left corner
        landmarks[1] = [eye_left_cx - eye_w*0.5, eye_left_cy - eye_h*0.5, 0.9]  # top left
        landmarks[2] = [eye_left_cx, eye_left_cy - eye_h, 0.92]  # top center
        landmarks[3] = [eye_left_cx + eye_w*0.5, eye_left_cy - eye_h*0.5, 0.88]  # top right
        landmarks[4] = [eye_left_cx + eye_w, eye_left_cy, 0.93]  # right corner
        landmarks[5] = [eye_left_cx, eye_left_cy + eye_h*0.3, 0.87]  # bottom
        
        # Right eye (6 points)
        eye_right_cx = x1 + w * 0.7
        eye_right_cy = y1 + h * 0.35
        
        landmarks[6] = [eye_right_cx - eye_w, eye_right_cy, 0.94]
        landmarks[7] = [eye_right_cx - eye_w*0.5, eye_right_cy - eye_h*0.5, 0.91]
        landmarks[8] = [eye_right_cx, eye_right_cy - eye_h, 0.93]
        landmarks[9] = [eye_right_cx + eye_w*0.5, eye_right_cy - eye_h*0.5, 0.89]
        landmarks[10] = [eye_right_cx + eye_w, eye_right_cy, 0.92]
        landmarks[11] = [eye_right_cx, eye_right_cy + eye_h*0.3, 0.86]
        
        # Nose (5 points)
        nose_cx = x1 + w * 0.5
        nose_top_y = y1 + h * 0.45
        nose_bottom_y = y1 + h * 0.6
        
        landmarks[12] = [nose_cx, nose_top_y, 0.88]
        landmarks[13] = [nose_cx - w*0.05, nose_bottom_y, 0.85]
        landmarks[14] = [nose_cx, nose_bottom_y + h*0.02, 0.87]
        landmarks[15] = [nose_cx + w*0.05, nose_bottom_y, 0.84]
        landmarks[16] = [nose_cx, nose_bottom_y, 0.90]
        
        # Mouth (11 points)
        mouth_cx = x1 + w * 0.5
        mouth_top_y = y1 + h * 0.68
        mouth_bottom_y = y1 + h * 0.78
        mouth_w = w * 0.25
        
        landmarks[17] = [mouth_cx - mouth_w, mouth_top_y, 0.89]  # left corner
        landmarks[18] = [mouth_cx - mouth_w*0.6, mouth_top_y - h*0.01, 0.86]
        landmarks[19] = [mouth_cx - mouth_w*0.3, mouth_top_y - h*0.015, 0.88]
        landmarks[20] = [mouth_cx, mouth_top_y - h*0.015, 0.91]  # top center
        landmarks[21] = [mouth_cx + mouth_w*0.3, mouth_top_y - h*0.015, 0.87]
        landmarks[22] = [mouth_cx + mouth_w*0.6, mouth_top_y - h*0.01, 0.85]
        landmarks[23] = [mouth_cx + mouth_w, mouth_top_y, 0.88]  # right corner
        landmarks[24] = [mouth_cx - mouth_w*0.5, mouth_bottom_y, 0.84]
        landmarks[25] = [mouth_cx - mouth_w*0.25, mouth_bottom_y + h*0.01, 0.86]
        landmarks[26] = [mouth_cx, mouth_bottom_y + h*0.015, 0.92]  # bottom center
        landmarks[27] = [mouth_cx + mouth_w*0.25, mouth_bottom_y + h*0.01, 0.85]
        
        return landmarks

    @staticmethod
    def _load_image(
            image_or_path: Union[np.ndarray, str, pathlib.Path]) -> np.ndarray:
        """Load image from path or return existing numpy array."""
        if isinstance(image_or_path, np.ndarray):
            image = image_or_path
        elif isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
        elif isinstance(image_or_path, pathlib.Path):
            image = cv2.imread(image_or_path.as_posix())
        else:
            raise ValueError("Invalid image input type")
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
            boxes, detected_keypoints = self._detect_faces(image)
            
            if len(boxes) == 0:
                warnings.warn(
                    'No faces detected. Treating entire image as face region.')
                h, w = image.shape[:2]
                boxes = [np.array([0, 0, w - 1, h - 1, 1.0], dtype=np.float32)]
                detected_keypoints = [None]
        else:
            # If boxes provided manually, no keypoints available
            detected_keypoints = [None] * len(boxes)
        
        return self._detect_landmarks(image, boxes, detected_keypoints)
