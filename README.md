# Human Pose Compare - macOS Version 1.2

This project compares human poses in videos using MediaPipe for pose detection and a simple Euclidean distance-based similarity metric. It is optimized for Python 3.12 on macOS and does **not** require TensorFlow or any legacy dependencies.

## Features
- Extracts 17 keypoints per frame using MediaPipe
- Compares videos against stored reference actions
- Supports multiple actions in a single lookup table
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

### 1. Create Reference Actions
Extract pose keypoints from reference videos and save as actions:

```bash
# Create a reference action
python3 keypoints_from_video.py --video punch_video.mp4 --action "punch" --lookup actions.pickle

# Add more actions to the same lookup table
python3 keypoints_from_video.py --video kick_video.mp4 --action "kick" --lookup actions.pickle
python3 keypoints_from_video.py --video jump_video.mp4 --action "jump" --lookup actions.pickle
```

### 2. Compare a Video Against Actions

#### Compare against a specific action:
```bash
python3 start_here.py --video test.mp4 --lookup actions.pickle --action "punch"
```

#### Compare against all available actions:
```bash
python3 start_here.py --video test.mp4 --lookup actions.pickle --all
```

#### Simple comparison (backward compatibility):
```bash
python3 start_here.py --video test.mp4 --lookup lookup.pickle
```

### Output
- Prints similarity scores (0-100%) and match status
- Shows pose mapping visualization
- When comparing against all actions, ranks them by similarity

## Example Workflow

1. **Create reference actions:**
   ```bash
   python3 keypoints_from_video.py --video punch.mp4 --action "punch" --lookup actions.pickle
   python3 keypoints_from_video.py --video kick.mp4 --action "kick" --lookup actions.pickle
   ```

2. **Test a new video:**
   ```bash
   python3 start_here.py --video test.mp4 --lookup actions.pickle --all
   ```

3. **Example output:**
   ```
   === Results (All Actions) ===
   punch: 85.23%
   kick: 12.45%
   
   Best match: punch (85.23%)
   ```

## File Structure
- `pose.py` — MediaPipe-based pose extraction
- `keypoints_from_video.py` — Extracts keypoints and creates action lookup tables
- `start_here.py` — Compares videos against actions
- `requirements.txt` — Only modern, minimal dependencies
- `actions.pickle` — Generated action lookup table
- `test.mp4` — Example video (replace with your own)

## No TensorFlow, No Legacy Code
All TensorFlow, posenet, and legacy files have been removed. This repo is now clean and modern.

## Troubleshooting
- Make sure you are using Python 3.12 and have activated your virtual environment.
- If you encounter MediaPipe or OpenCV errors, ensure your dependencies are up to date.
- Use `--no-viz` flag to disable visualization if you don't need it.

## License
MIT
