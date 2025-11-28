import pathlib

import torch

from .detector import LandmarkDetector

# Try to import mmdet-based detector
try:
    from .mmdet_detector import MMDetLandmarkDetector
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False
    MMDetLandmarkDetector = None


def get_model_path(model_name: str) -> pathlib.Path:
    """
    Get or download YOLO model file.
    
    Args:
        model_name: Name of the model ('yolov8n', 'yolov8s', 'yolov8m', etc.)
        
    Returns:
        Path to the model file
    """
    # Use ultralytics default model location
    # Models will be auto-downloaded by ultralytics on first use
    model_file = f'{model_name}.pt'
    return pathlib.Path(model_file)


def create_detector(face_detector_name: str = 'yolov8n',
                    device: str = 'cuda:0',
                    box_scale_factor: float = 1.1,
                    confidence_threshold: float = 0.25,
                    use_mmdet: bool = False) -> LandmarkDetector:
    """
    Create an anime face landmark detector.
    
    Args:
        face_detector_name: Model name
                          - For ultralytics: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
                          - For mmdet (use_mmdet=True): 'yolov3', 'faster-rcnn'
        device: Device to run on ('cuda:0', 'cuda:1', 'cpu', etc.)
        box_scale_factor: Scale factor for bounding boxes (default: 1.1)
        confidence_threshold: Minimum confidence for face detection (default: 0.25)
        use_mmdet: If True, use original MMDetection-based detector with anime face models
                   (requires mmdet/mmpose installation)
        
    Returns:
        Initialized LandmarkDetector instance
        
    Note:
        Models will be automatically downloaded on first use.
        For Windows compatibility, 'cpu' device is recommended if CUDA is not available.
        
        For better anime face landmark detection, set use_mmdet=True and install:
          pip install openmim
          mim install mmcv>=2.0.0
          pip install mmdet>=3.0.0 mmpose>=1.0.0
    """
    # Ensure device is valid
    if not torch.cuda.is_available() and 'cuda' in device:
        print(f"CUDA not available, falling back to CPU")
        device = 'cpu'
    
    if use_mmdet:
        # Use original MMDetection-based detector
        if not MMDET_AVAILABLE:
            raise ImportError(
                "MMDetection/MMPose not available. Please install:\n"
                "  pip install openmim\n"
                "  mim install mmcv>=2.0.0\n"
                "  pip install mmdet>=3.0.0 mmpose>=1.0.0"
            )
        
        # Map detector names
        detector_mapping = {
            'yolov3': 'yolov3',
            'faster-rcnn': 'faster-rcnn',
            'yolov8n': 'yolov3',
            'yolov8s': 'faster-rcnn',
        }
        
        face_detector = detector_mapping.get(face_detector_name, 'yolov3')
        
        detector = MMDetLandmarkDetector(
            face_detector=face_detector,
            landmark_model='hrnetv2',
            device=device,
            box_scale_factor=box_scale_factor,
            face_score_threshold=confidence_threshold
        )
    else:
        # Use ultralytics-based detector (Windows-compatible)
        # Map old detector names to new YOLO pose models for better landmark detection
        detector_mapping = {
            'yolov3': 'yolov8n-pose',  # lightweight with keypoints
            'faster-rcnn': 'yolov8s-pose',  # more accurate with keypoints
            'yolov8n': 'yolov8n-pose',
            'yolov8s': 'yolov8s-pose',
            'yolov8m': 'yolov8m-pose',
            'yolov8l': 'yolov8l-pose',
            'yolov8x': 'yolov8x-pose',
        }
        
        model_name = detector_mapping.get(face_detector_name, face_detector_name)
        
        # Add -pose suffix if not present and not a custom path
        if not model_name.endswith('.pt') and 'pose' not in model_name:
            model_name = f'{model_name}-pose'
        
        model_path = get_model_path(model_name) if not pathlib.Path(model_name).exists() else pathlib.Path(model_name)
        
        detector = LandmarkDetector(
            face_detector_model=model_path.as_posix(),
            device=device,
            box_scale_factor=box_scale_factor,
            confidence_threshold=confidence_threshold
        )
    
    return detector


__all__ = ['LandmarkDetector', 'MMDetLandmarkDetector', 'create_detector', 'MMDET_AVAILABLE']
