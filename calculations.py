import cv2
import numpy as np
import pickle
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from pose import PoseDetector

def calculate_similarity(pose1, pose2):
    """
    Calculate similarity between two pose sequences using FastDTW and cosine similarity
    Args:
        pose1: First pose sequence (numpy array)
        pose2: Second pose sequence (numpy array)
    Returns:
        Similarity score (0-1, where 1 is identical)
    """
    # Ensure poses are properly shaped for DTW
    if len(pose1.shape) == 3:  # (frames, 33, 3)
        pose1_2d = pose1.reshape(pose1.shape[0], -1)  # Flatten to (frames, 99)
    else:  # Already flattened
        pose1_2d = pose1.reshape(-1, 99) if pose1.shape[-1] == 99 else pose1
    
    if len(pose2.shape) == 3:  # (frames, 33, 3)
        pose2_2d = pose2.reshape(pose2.shape[0], -1)  # Flatten to (frames, 99)
    else:  # Already flattened
        pose2_2d = pose2.reshape(-1, 99) if pose2.shape[-1] == 99 else pose2
    
    # Ensure arrays are finite and not empty
    if pose1_2d.size == 0 or pose2_2d.size == 0:
        return 0.0
    
    # Remove any NaN or infinite values
    pose1_2d = pose1_2d[np.isfinite(pose1_2d).all(axis=1)]
    pose2_2d = pose2_2d[np.isfinite(pose2_2d).all(axis=1)]
    
    if pose1_2d.size == 0 or pose2_2d.size == 0:
        return 0.0
    
    try:
        # Use FastDTW for better performance and accuracy
        distance, path = fastdtw(pose1_2d, pose2_2d, dist=euclidean)
        
        # Calculate cosine similarity for DTW-selected frame pairs
        pose_detector = PoseDetector()
        similarities = []
        
        for i, j in path:
            if i < len(pose1_2d) and j < len(pose2_2d):
                sim = pose_detector.cosine_similarity_pose(
                    pose1_2d[i].reshape(-1, 3),
                    pose2_2d[j].reshape(-1, 3)
                )
                similarities.append(sim)
        
        # Return average similarity
        if similarities:
            avg_similarity = np.mean(similarities)
            return max(0, avg_similarity)  # Ensure non-negative
        else:
            return 0.0
            
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
            
        # Extract pose keypoints (now returns 99-element array)
        keypoints = pose_detector.getpoints(frame)
        poses.append(keypoints)
    
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

class PoseScorer:
    """
    Modern pose scoring class using MediaPipe and FastDTW
    """
    def __init__(self, lookup_path='lookup.pickle'):
        self.pose_detector = PoseDetector()
        self.lookup_data = self.load_lookup(lookup_path)
    
    def load_lookup(self, lookup_path):
        """Load lookup table from pickle file"""
        try:
            with open(lookup_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Lookup file not found: {lookup_path}")
            return None
    
    def get_action_coords_from_dict(self, action):
        """Get pose data for specific action"""
        if self.lookup_data is None:
            return None, 0
        
        if action in self.lookup_data:
            model_array = self.lookup_data[action]
            no_of_frames = len(model_array)
            return model_array, no_of_frames
        else:
            print(f"Action '{action}' not found in lookup table")
            return None, 0
    
    def calculate_score(self, video_path, action):
        """
        Calculate similarity score between video and stored action
        Args:
            video_path: Path to input video
            action: Action name to compare against
        Returns:
            Tuple of (final_score, score_list)
        """
        # Get reference pose data
        model_array, num_frames = self.get_action_coords_from_dict(action)
        if model_array is None:
            return 0.0, []
        
        # Process input video
        input_poses = process_video(video_path, self.pose_detector)
        
        # Calculate similarity
        similarity = calculate_similarity(input_poses, model_array)
        
        # Create score list (simplified for compatibility)
        score_list = [similarity] * len(input_poses)
        
        return similarity, score_list 