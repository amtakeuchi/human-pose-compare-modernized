import cv2
import numpy as np
import pickle
from pose import PoseDetector

def calculate_similarity(pose1, pose2):
    """
    Calculate similarity between two pose sequences using Euclidean distance
    Args:
        pose1: First pose sequence (numpy array)
        pose2: Second pose sequence (numpy array)
    Returns:
        Similarity score (0-1, where 1 is identical)
    """
    # Ensure poses are 2D arrays
    if len(pose1.shape) == 3:  # (frames, 17, 2)
        pose1_2d = pose1.reshape(-1, 2)
    else:  # (frames * 17, 2)
        pose1_2d = pose1.reshape(-1, 2)
    
    if len(pose2.shape) == 3:  # (frames, 17, 2)
        pose2_2d = pose2.reshape(-1, 2)
    else:  # (frames * 17, 2)
        pose2_2d = pose2.reshape(-1, 2)
    
    # Ensure arrays are finite and not empty
    if pose1_2d.size == 0 or pose2_2d.size == 0:
        return 0.0
    
    # Remove any NaN or infinite values
    pose1_2d = pose1_2d[np.isfinite(pose1_2d).all(axis=1)]
    pose2_2d = pose2_2d[np.isfinite(pose2_2d).all(axis=1)]
    
    if pose1_2d.size == 0 or pose2_2d.size == 0:
        return 0.0
    
    try:
        # Calculate average Euclidean distance between corresponding points
        # Use the shorter sequence length to avoid index errors
        min_length = min(len(pose1_2d), len(pose2_2d))
        
        if min_length == 0:
            return 0.0
        
        # Calculate distances for each corresponding point
        distances = []
        for i in range(min_length):
            dist = np.linalg.norm(pose1_2d[i] - pose2_2d[i])
            distances.append(dist)
        
        # Calculate average distance
        avg_distance = np.mean(distances)
        
        # Normalize to similarity score (0-1)
        # Assuming maximum reasonable distance is 1000 pixels
        max_distance = 1000.0
        similarity = max(0, 1 - (avg_distance / max_distance))
        
        return similarity
    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0.0

def process_video(video_path, pose_detector):
    """
    Process video and extract pose keypoints
    Args:
        video_path: Path to video file
        pose_detector: PoseDetector instance
    Returns:
        List of pose keypoints for each frame
    """
    cap = cv2.VideoCapture(video_path)
    poses = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract pose keypoints
        keypoints = pose_detector.getpoints(frame)
        # Reshape to 17x2 array
        pose_2d = keypoints.reshape(17, 2)
        poses.append(pose_2d)
    
    cap.release()
    return np.array(poses)

def compare_poses(video1_path, video2_path):
    """
    Compare two videos and return similarity score
    Args:
        video1_path: Path to first video
        video2_path: Path to second video
    Returns:
        Similarity score (0-1)
    """
    pose_detector = PoseDetector()
    
    # Process both videos
    poses1 = process_video(video1_path, pose_detector)
    poses2 = process_video(video2_path, pose_detector)
    
    # Calculate similarity
    similarity = calculate_similarity(poses1, poses2)
    
    return similarity 