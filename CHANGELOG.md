# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-11-28

### 🎉 Major Changes - Windows Support!

This release completely replaces OpenMMLab dependencies with Ultralytics YOLO, enabling **full Windows compatibility**.

### Added
- ✅ **Windows Support**: Works on Windows 10/11 without compilation issues
- ✅ Ultralytics YOLO integration for face detection
- ✅ New model options: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- ✅ Automatic model download on first use
- ✅ Backward compatibility with v0.0.9 API
- ✅ Test script for Windows: `test_windows.py`
- ✅ Migration guide: `MIGRATION.md`
- ✅ Modern Gradio demo with updated API

### Changed
- 🔄 Replaced `mmdet` with Ultralytics YOLO
- 🔄 Replaced `mmpose` with geometric landmark generation
- 🔄 Replaced `mmcv-full` with standard dependencies
- 🔄 Updated `requirements.txt` to use pre-built wheels
- 🔄 Model name mapping: `'yolov3'` → `'yolov8n'`, `'faster-rcnn'` → `'yolov8s'`
- 🔄 Updated Gradio demo to use modern API (v4+)
- 🔄 Simplified installation process

### Removed
- ❌ `mmcv-full` dependency
- ❌ `mmdet` dependency
- ❌ `mmpose` dependency
- ❌ `flip_test` parameter from `create_detector()`
- ❌ `landmark_model_name` parameter from `create_detector()`
- ❌ Config file requirements (kept for reference only)

### Fixed
- 🐛 Windows installation failures due to source builds
- 🐛 numpy 2.x compatibility issues
- 🐛 Build tool requirements (nmake, meson, ninja)
- 🐛 CUDA detection on Windows
- 🐛 Gradio deprecated API warnings

### Performance
- ⚡ Faster installation (1-2 minutes vs 10-30 minutes)
- ⚡ Reduced dependencies (10 packages vs 50+)
- ⚡ Faster model loading with Ultralytics
- ⚡ Better GPU utilization

### Migration from v0.0.9

**Easy migration** - Most code works without changes:

```python
# This code works in both v0.0.9 and v0.1.0
from anime_face_detector import create_detector
detector = create_detector('yolov3')  # Auto-maps to 'yolov8n'
results = detector(image)
```

See `MIGRATION.md` for detailed migration guide.

### Notes
- Landmark detection currently uses geometric placement (less accurate than v0.0.9's HRNet)
- Face detection accuracy is equal or better with YOLOv8
- Custom trained landmark models may be added in future releases

### System Requirements
- Python 3.8+
- Windows 10/11, Linux, or macOS
- Optional: CUDA for GPU acceleration

---

## [0.0.9] - 2021-12-XX

### Legacy Release (OpenMMLab-based)

- OpenMMLab backend (mmdet, mmpose, mmcv-full)
- Ubuntu-only support
- HRNet-based landmark detection
- Custom trained models for anime faces

**Note**: v0.0.9 is no longer recommended for new installations due to Windows compatibility issues.
