import pathlib

import torch

from .detector import LandmarkDetector


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
                    confidence_threshold: float = 0.25) -> LandmarkDetector:
    """
    Create an anime face landmark detector.
    
    Args:
        face_detector_name: YOLO model name ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
                          or custom model path. Smaller models are faster but less accurate.
        device: Device to run on ('cuda:0', 'cuda:1', 'cpu', etc.)
        box_scale_factor: Scale factor for bounding boxes (default: 1.1)
        confidence_threshold: Minimum confidence for face detection (default: 0.25)
        
    Returns:
        Initialized LandmarkDetector instance
        
    Note:
        Models will be automatically downloaded on first use.
        For Windows compatibility, 'cpu' device is recommended if CUDA is not available.
    """
    # Map old detector names to new YOLO models
    detector_mapping = {
        'yolov3': 'yolov8n',  # lightweight
        'faster-rcnn': 'yolov8s',  # more accurate
        'yolov8n': 'yolov8n',
        'yolov8s': 'yolov8s',
        'yolov8m': 'yolov8m',
        'yolov8l': 'yolov8l',
        'yolov8x': 'yolov8x',
    }
    
    model_name = detector_mapping.get(face_detector_name, face_detector_name)
    model_path = get_model_path(model_name)
    
    # Ensure device is valid
    if not torch.cuda.is_available() and 'cuda' in device:
        print(f"CUDA not available, falling back to CPU")
        device = 'cpu'
    
    detector = LandmarkDetector(
        face_detector_model=model_path.as_posix(),
        device=device,
        box_scale_factor=box_scale_factor,
        confidence_threshold=confidence_threshold
    )
    
    return detector


__all__ = ['LandmarkDetector', 'create_detector']
