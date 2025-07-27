# Human Pose Compare - Modernized

A modernized version of human pose comparison and action scoring using computer vision. This project compares video sequences to detect and score human poses using OpenCV and Dynamic Time Warping (DTW).

## 🚀 Features

- **Modern Python Compatibility**: Works with Python 3.8+ and current package versions
- **No TensorFlow Dependency**: Uses OpenCV for pose detection instead of TensorFlow 1.x
- **Real-time Processing**: Extract pose keypoints from video frames
- **Action Scoring**: Compare videos against stored pose sequences
- **DTW Algorithm**: Uses Dynamic Time Warping for robust pose comparison
- **Easy Setup**: Simple installation with minimal dependencies

## 📋 Requirements

- Python 3.8+
- OpenCV
- NumPy
- SciPy
- dtaidistance (for DTW)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/human-pose-compare-modernized.git
   cd human-pose-compare-modernized
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements_simple.txt
   ```

## 🎯 Usage

### 1. Extract Keypoints from a Video

Create a lookup table from a reference video:

```bash
python keypoints_from_video_simple.py --activity "punch - side" --video "test.mp4" --lookup "my_lookup.pickle"
```

**Parameters:**
- `--activity`: Name of the action/pose sequence
- `--video`: Path to the input video file
- `--lookup`: Output pickle file to store the keypoints

### 2. Compare a Video Against a Lookup

Score a video against a stored action:

```bash
python start_here_simple.py --activity "punch - side" --video "test.mp4" --lookup "lookup_test.pickle"
```

**Parameters:**
- `--activity`: Name of the action to compare against
- `--video`: Path to the video file to score
- `--lookup`: Path to the pickle file containing reference keypoints

## 📊 Example Output

```
Total Score :  99.47
Score List :  [99, 100, 99, 100, 99, 100, 99, 99, 100, 99, 100, 100, 100, 99, 99, 99, 100]
```

- **Total Score**: Average similarity score (0-100, higher is better)
- **Score List**: Individual scores for each of the 17 keypoints

## 🏗️ Architecture

### Core Components

1. **Pose Detection** (`pose_simple.py`)
   - Uses OpenCV contour detection
   - Extracts 17 keypoints per frame
   - Normalizes coordinates relative to bounding box

2. **Keypoint Extraction** (`keypoints_from_video_simple.py`)
   - Processes video frames sequentially
   - Saves keypoint sequences to pickle files
   - Creates lookup tables for comparison

3. **Pose Comparison** (`calculations_simple.py`)
   - Implements DTW algorithm for sequence comparison
   - Calculates similarity scores
   - Handles different frame counts

4. **Scoring System** (`score.py`)
   - Normalizes keypoint coordinates
   - Applies DTW distance calculation
   - Converts distances to percentage scores

## 📁 Project Structure

```
human-pose-compare-modernized/
├── pose_simple.py                    # Simplified pose detection
├── calculations_simple.py            # Updated calculations module
├── keypoints_from_video_simple.py   # Keypoints extraction script
├── start_here_simple.py             # Main comparison script
├── score.py                         # Scoring algorithms
├── requirements_simple.txt          # Modern dependencies
├── README.md                        # This file
├── test.mp4                         # Sample video (not included in repo)
├── lookup_test.pickle              # Sample lookup file
└── venv/                           # Virtual environment (not tracked)
```

## 🔧 How It Works

1. **Video Processing**: Reads video frames and resizes to standard dimensions
2. **Pose Detection**: Uses OpenCV contour detection to identify human shapes
3. **Keypoint Generation**: Creates 17 keypoints distributed across the detected person
4. **ROI Processing**: Normalizes coordinates relative to the person's bounding box
5. **DTW Comparison**: Uses Dynamic Time Warping to compare pose sequences
6. **Scoring**: Returns percentage scores based on pose similarity

## ⚠️ Limitations

- **Simplified Detection**: Basic contour-based detection (not as accurate as deep learning)
- **Keypoint Distribution**: Keypoints are distributed geometrically rather than anatomically
- **No Advanced Features**: Missing confidence scoring and advanced pose estimation

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

2. **"No module named 'numpy'"**
   ```bash
   pip install numpy
   ```

3. **Video not found**
   - Check file path is correct
   - Ensure video file exists and is readable

4. **Lookup file not found**
   - Create a lookup file first using the keypoints extraction script
   - Check file path and permissions

### Performance Tips

- Use shorter videos for faster processing
- Ensure good lighting and clear video quality
- Results may vary based on video complexity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original implementation: [Human-Pose-Compare](https://github.com/krishnarajr319/Human-Pose-Compare)
- DTW implementation: [dtaidistance](https://github.com/wannesm/dtaidistance)
- OpenCV for computer vision capabilities

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Note**: This is a modernized version of the original Human Pose Compare project, updated to work with current Python versions and modern dependencies.
