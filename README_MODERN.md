# Human Pose Compare - Modernized Version

This is a modernized version of the Human Pose Compare project that works with current Python versions and doesn't require TensorFlow 1.x or other outdated dependencies.

## What's Changed

- **Removed TensorFlow 1.x dependency**: The original code used TensorFlow 1.x with `tf.Session()` which is deprecated
- **Replaced PoseNet with OpenCV-based detection**: Uses OpenCV's contour detection for pose estimation
- **Updated dependencies**: Uses modern versions of OpenCV, NumPy, and SciPy
- **Simplified installation**: No complex model downloads required

## Installation

1. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. **Install dependencies**:
```bash
pip install -r requirements_simple.txt
```

## Usage

### 1. Extract Keypoints from a Video

To create a new lookup table from a video:

```bash
python keypoints_from_video_simple.py --activity "punch - side" --video "test.mp4" --lookup "my_lookup.pickle"
```

This will:
- Process each frame of the video
- Extract pose keypoints using OpenCV contour detection
- Save the keypoints to a pickle file

### 2. Compare a Video Against a Lookup

To score a video against a stored action:

```bash
python start_here_simple.py --activity "punch - side" --video "test.mp4" --lookup "lookup_test.pickle"
```

This will:
- Process the input video
- Compare it against the stored action
- Return a score (higher is better) and individual keypoint scores

## Files Overview

### Modernized Files
- `pose_simple.py`: Simplified pose detection using OpenCV
- `calculations_simple.py`: Updated calculations module
- `keypoints_from_video_simple.py`: Script to extract keypoints from videos
- `start_here_simple.py`: Main script for pose comparison
- `requirements_simple.txt`: Modern dependencies

### Original Files (for reference)
- `pose.py`: Original TensorFlow-based pose detection
- `calculations.py`: Original calculations module
- `keypoints_from_video.py`: Original keypoints extraction
- `start_here.py`: Original main script

## How It Works

1. **Pose Detection**: Uses OpenCV contour detection to identify human-like shapes in video frames
2. **Keypoint Generation**: Creates 17 keypoints distributed across the detected person
3. **ROI Processing**: Normalizes coordinates relative to the person's bounding box
4. **DTW Comparison**: Uses Dynamic Time Warping to compare pose sequences
5. **Scoring**: Returns a percentage score based on pose similarity

## Limitations

- **Simplified Detection**: The pose detection is basic and may not be as accurate as the original TensorFlow-based system
- **Keypoint Distribution**: Keypoints are distributed based on bounding box rather than actual body parts
- **No Advanced Features**: Missing advanced features like pose confidence scoring

## Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**: Install OpenCV: `pip install opencv-python`
2. **"No module named 'numpy'"**: Install NumPy: `pip install numpy`
3. **Video not found**: Make sure the video file path is correct
4. **Lookup file not found**: Create a lookup file first using the keypoints extraction script

### Performance Tips

- Use shorter videos for faster processing
- The system works best with clear, well-lit videos
- Results may vary depending on video quality and pose complexity

## Example Output

```
Total Score :  99.47
Score List :  [99, 100, 99, 100, 99, 100, 99, 99, 100, 99, 100, 100, 100, 99, 99, 99, 100]
```

The total score is the average of individual keypoint scores, where 100 is perfect similarity.

## Compatibility

This modernized version is compatible with:
- Python 3.8+
- macOS, Linux, and Windows
- Modern OpenCV, NumPy, and SciPy versions

The system maintains compatibility with existing lookup files from the original implementation. 