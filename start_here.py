#!/usr/bin/env python3
"""
Human Pose Comparison Tool - macOS Version 1.2
Uses MediaPipe for pose detection and Euclidean similarity
"""

import cv2
import numpy as np
import pickle
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

def compare_with_lookup(video_path, lookup_path, show_visualization=True):
    lookup_data = load_lookup_table(lookup_path)
    if lookup_data is None:
        return 0.0
    pose_detector = PoseDetector()
    print("Processing input video...")
    input_poses = process_video(video_path, pose_detector)
    lookup_poses = []
    for frame_num in sorted(lookup_data.keys()):
        keypoints = lookup_data[frame_num]
        pose_2d = keypoints.reshape(17, 2)
        lookup_poses.append(pose_2d)
    lookup_poses = np.array(lookup_poses)
    print("Calculating similarity...")
    similarity = calculate_similarity(input_poses, lookup_poses)
    if show_visualization:
        print("Showing pose mapping visualization. Press 'q' to quit.")
        visualize_pose_on_video(video_path, pose_detector)
    return similarity

def main():
    print("=== Human Pose Comparison Tool - macOS Version 1.2 ===")
    print("Using MediaPipe for pose detection")
    print()
    video_file = "test.mp4"
    lookup_file = "lookup.pickle"
    print(f"Input video: {video_file}")
    print(f"Lookup table: {lookup_file}")
    print()
    try:
        similarity_score = compare_with_lookup(video_file, lookup_file, show_visualization=True)
        print(f"\n=== Results ===")
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