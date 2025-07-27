# Human Pose Compare - macOS Version 1.2

This project compares human poses in videos using MediaPipe for pose detection and a simple Euclidean distance-based similarity metric. It is optimized for Python 3.12 on macOS and does **not** require TensorFlow or any legacy dependencies.

## Features
- Extracts 17 keypoints per frame using MediaPipe
- Compares two videos (or a video and a reference) for pose similarity
- Simple, modern, and easy to use

## Requirements
- Python 3.12 (recommended)
- macOS (Apple Silicon or Intel)
- [MediaPipe](https://google.github.io/mediapipe/solutions/pose.html)
- OpenCV
- NumPy

Install dependencies:
```bash
python3 -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Extract Keypoints and Create Lookup Table
Extract pose keypoints from a reference video and save as a lookup table:
```bash
python3 keypoints_from_video.py
```
- This will process `test.mp4` and create `lookup.pickle` by default.
- You can edit the script to use your own video file.

### 2. Compare a New Video to the Lookup Table
Compare a new video to the reference lookup table:
```bash
python3 start_here.py
```
- By default, compares `test.mp4` to `lookup.pickle`.
- Edit `start_here.py` to use your own files if needed.

### Output
- Prints a similarity score (0-100%) and a match status.

## File Structure
- `pose.py` — MediaPipe-based pose extraction
- `keypoints_from_video.py` — Extracts keypoints and creates lookup table
- `start_here.py` — Compares a video to the lookup table
- `requirements.txt` — Only modern, minimal dependencies
- `lookup.pickle` — Generated lookup table (after running extraction)
- `test.mp4` — Example video (replace with your own)

## No TensorFlow, No Legacy Code
All TensorFlow, posenet, and legacy files have been removed. This repo is now clean and modern.

## Troubleshooting
- Make sure you are using Python 3.12 and have activated your virtual environment.
- If you encounter MediaPipe or OpenCV errors, ensure your dependencies are up to date.

## License
MIT
