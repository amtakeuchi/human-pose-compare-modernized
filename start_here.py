#!/usr/bin/env python3
"""
Human Pose Comparison Tool - macOS Version 1.2
Uses MediaPipe for pose detection and FastDTW with cosine similarity
"""

import cv2
import numpy as np
import pickle
import argparse
from pose import PoseDetector
from calculations import calculate_similarity, process_video

def load_lookup_table(lookup_path):
    try:
        with open(lookup_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Lookup file not found: {lookup_path}")
        return None

def visualize_pose_on_video(video_path, pose_detector):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints, annotated = pose_detector.getpoints_vis(frame)
        cv2.imshow('Pose Mapping', annotated)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f"Displayed {frame_count} frames with pose mapping.")

def compare_with_lookup(video_path, lookup_path, action_name=None, show_visualization=True):
    lookup_data = load_lookup_table(lookup_path)
    if lookup_data is None:
        return 0.0, None
    
    pose_detector = PoseDetector()
    print("Processing input video...")
    input_poses = process_video(video_path, pose_detector)
    
    # Check if this is an action-based lookup table
    if action_name:
        if action_name not in lookup_data:
            print(f"Action '{action_name}' not found in lookup table")
            print(f"Available actions: {list(lookup_data.keys())}")
            return 0.0, None
        
        # Get the reference action data
        action_data = lookup_data[action_name]
        lookup_poses = []
        for frame_num in sorted(action_data.keys()):
            keypoints = action_data[frame_num]  # 99-element array (33 points * 3 coordinates)
            # Keep as 99-element array for the new format
            lookup_poses.append(keypoints)
        lookup_poses = np.array(lookup_poses)
        
        print(f"Comparing against action: {action_name}")
    else:
        # Simple lookup table (backward compatibility)
        lookup_poses = []
        for frame_num in sorted(lookup_data.keys()):
            keypoints = lookup_data[frame_num]
            # Handle both old (34-element) and new (99-element) formats
            if len(keypoints) == 34:
                # Old format: reshape to 17x2 array
                pose_2d = keypoints.reshape(17, 2)
                lookup_poses.append(pose_2d)
            else:
                # New format: keep as 99-element array
                lookup_poses.append(keypoints)
        lookup_poses = np.array(lookup_poses)
        print("Comparing against simple lookup table")
    
    print("Calculating similarity...")
    similarity = calculate_similarity(input_poses, lookup_poses)
    
    if show_visualization:
        print("Showing pose mapping visualization. Press 'q' to quit.")
        visualize_pose_on_video(video_path, pose_detector)
    
    return similarity, action_name

def compare_all_actions(video_path, lookup_path, show_visualization=False):
    """Compare video against all available actions"""
    lookup_data = load_lookup_table(lookup_path)
    if lookup_data is None:
        return {}
    
    pose_detector = PoseDetector()
    print("Processing input video...")
    input_poses = process_video(video_path, pose_detector)
    
    results = {}
    
    for action_name in lookup_data.keys():
        print(f"Comparing against action: {action_name}")
        action_data = lookup_data[action_name]
        lookup_poses = []
        for frame_num in sorted(action_data.keys()):
            keypoints = action_data[frame_num]
            # Handle both old (34-element) and new (99-element) formats
            if len(keypoints) == 34:
                # Old format: reshape to 17x2 array
                pose_2d = keypoints.reshape(17, 2)
                lookup_poses.append(pose_2d)
            else:
                # New format: keep as 99-element array
                lookup_poses.append(keypoints)
        lookup_poses = np.array(lookup_poses)
        
        similarity = calculate_similarity(input_poses, lookup_poses)
        results[action_name] = similarity
    
    if show_visualization:
        print("Showing pose mapping visualization. Press 'q' to quit.")
        visualize_pose_on_video(video_path, pose_detector)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare video poses')
    parser.add_argument('--video', default='test.mp4', help='Input video file')
    parser.add_argument('--lookup', default='lookup.pickle', help='Lookup table file')
    parser.add_argument('--action', help='Specific action to compare against')
    parser.add_argument('--all', action='store_true', help='Compare against all actions')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    print("=== Human Pose Comparison Tool - macOS Version 1.2 ===")
    print("Using MediaPipe for pose detection with FastDTW and cosine similarity")
    print()
    print(f"Input video: {args.video}")
    print(f"Lookup table: {args.lookup}")
    print()
    
    try:
        if args.all:
            # Compare against all actions
            results = compare_all_actions(args.video, args.lookup, not args.no_viz)
            print(f"\n=== Results (All Actions) ===")
            for action, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
                print(f"{action}: {score:.2%}")
            
            best_action = max(results.items(), key=lambda x: x[1])
            print(f"\nBest match: {best_action[0]} ({best_action[1]:.2%})")
            
        else:
            # Compare against specific action or simple lookup
            similarity_score, action_name = compare_with_lookup(
                args.video, args.lookup, args.action, not args.no_viz
            )
            
            print(f"\n=== Results ===")
            if action_name:
                print(f"Action: {action_name}")
            print(f"Similarity Score: {similarity_score:.2%}")
            
            if similarity_score > 0.8:
                print("Status: Excellent match!")
            elif similarity_score > 0.6:
                print("Status: Good match")
            elif similarity_score > 0.4:
                print("Status: Moderate match")
            else:
                print("Status: Poor match")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure the video file and lookup table exist.")

if __name__ == "__main__":
    main()