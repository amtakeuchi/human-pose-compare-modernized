import cv2
import numpy as np
import pickle
from pose import PoseDetector

def extract_keypoints_from_video(video_path, output_path=None):
    """
    Extract pose keypoints from video and save to pickle file
    Args:
        video_path: Path to input video
        output_path: Path for output pickle file (optional)
    Returns:
        Dictionary with frame keypoints
    """
    pose_detector = PoseDetector()
    cap = cv2.VideoCapture(video_path)
    
    frame_keypoints = {}
    frame_count = 0
    
    print(f"Processing video: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract pose keypoints
        keypoints = pose_detector.getpoints(frame)
        
        # Store keypoints for this frame
        frame_keypoints[frame_count] = keypoints
        
        frame_count += 1
        
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    
    print(f"Total frames processed: {frame_count}")
    
    # Save to pickle file if output path provided
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(frame_keypoints, f)
        print(f"Keypoints saved to: {output_path}")
    
    return frame_keypoints

def create_lookup_table(video_path, output_path="lookup.pickle"):
    """
    Create a lookup table from video keypoints
    Args:
        video_path: Path to video file
        output_path: Output pickle file path
    """
    keypoints = extract_keypoints_from_video(video_path, output_path)
    return keypoints

if __name__ == "__main__":
    # Example usage
    video_file = "test.mp4"  # Change to your video file
    lookup_file = "lookup.pickle"
    
    try:
        keypoints = create_lookup_table(video_file, lookup_file)
        print(f"Successfully created lookup table with {len(keypoints)} frames")
    except Exception as e:
        print(f"Error processing video: {e}")


