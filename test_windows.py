"""
Simple test script to verify anime-face-detector works on Windows
"""
import sys
import platform

print("=" * 60)
print("Anime Face Detector - Windows Compatibility Test")
print("=" * 60)
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print()

# Test imports
print("Testing imports...")
try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except ImportError as e:
    print(f"✗ numpy failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✓ opencv-python {cv2.__version__}")
except ImportError as e:
    print(f"✗ opencv-python failed: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✓ torch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
except ImportError as e:
    print(f"✗ torch failed: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print(f"✓ ultralytics imported successfully")
except ImportError as e:
    print(f"✗ ultralytics failed: {e}")
    sys.exit(1)

try:
    from anime_face_detector import create_detector, LandmarkDetector
    print(f"✓ anime_face_detector imported successfully")
except ImportError as e:
    print(f"✗ anime_face_detector failed: {e}")
    sys.exit(1)

print()
print("All imports successful!")
print()

# Test detector creation
print("Testing detector creation...")
try:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    detector = create_detector('yolov8n', device=device)
    print(f"✓ Detector created successfully")
    print(f"  Model: yolov8n")
    print(f"  Device: {detector.device}")
    print(f"  Box scale factor: {detector.box_scale_factor}")
except Exception as e:
    print(f"✗ Detector creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Testing with dummy image...")
try:
    # Create a simple test image (anime-like face placeholder)
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Run detection
    results = detector(test_image)
    print(f"✓ Detection completed")
    print(f"  Number of detections: {len(results)}")
    
    if len(results) > 0:
        print(f"  First detection:")
        print(f"    - Bbox shape: {results[0]['bbox'].shape}")
        print(f"    - Keypoints shape: {results[0]['keypoints'].shape}")
except Exception as e:
    print(f"✗ Detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("✅ All tests passed! anime-face-detector is working on Windows!")
print("=" * 60)
print()
print("Next steps:")
print("1. Try with real anime images:")
print("   python -c \"from anime_face_detector import create_detector; import cv2; detector = create_detector('yolov8n'); img = cv2.imread('your_image.jpg'); print(detector(img))\"")
print()
print("2. Run the Gradio demo:")
print("   pip install gradio")
print("   python demo_gradio.py")
