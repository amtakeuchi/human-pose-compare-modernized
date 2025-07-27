import cv2
import numpy as np
import pickle
import argparse
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

def create_action_lookup_table(video_path, action_name, output_path="lookup.pickle"):
    """
    Create a lookup table for a specific action
    Args:
        video_path: Path to video file
        action_name: Name of the action (e.g., "punch", "kick", "jump")
        output_path: Output pickle file path
    """
    # Load existing lookup table if it exists
    try:
        with open(output_path, 'rb') as f:
            lookup_data = pickle.load(f)
    except FileNotFoundError:
        lookup_data = {}
    
    # Extract keypoints for this action
    keypoints = extract_keypoints_from_video(video_path)
    
    # Store under the action name
    lookup_data[action_name] = keypoints
    
    # Save updated lookup table
    with open(output_path, 'wb') as f:
        pickle.dump(lookup_data, f)
    
    print(f"Action '{action_name}' added to lookup table")
    return lookup_data

def create_simple_lookup_table(video_path, output_path="lookup.pickle"):
    """
    Create a simple lookup table from video keypoints (for backward compatibility)
    Args:
        video_path: Path to video file
        output_path: Output pickle file path
    """
    keypoints = extract_keypoints_from_video(video_path, output_path)
    return keypoints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract pose keypoints from video')
    parser.add_argument('--video', default='test.mp4', help='Input video file')
    parser.add_argument('--action', help='Action name (e.g., "punch", "kick")')
    parser.add_argument('--lookup', default='lookup.pickle', help='Output lookup file')
    
    args = parser.parse_args()
    
    try:
        if args.action:
            # Create action-based lookup table
            lookup_data = create_action_lookup_table(args.video, args.action, args.lookup)
            print(f"Successfully created action-based lookup table with {len(lookup_data)} actions")
        else:
            # Create simple lookup table (backward compatibility)
            keypoints = create_simple_lookup_table(args.video, args.lookup)
            print(f"Successfully created simple lookup table with {len(keypoints)} frames")
    except Exception as e:
        print(f"Error processing video: {e}")


