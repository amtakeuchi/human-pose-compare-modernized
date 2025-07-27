import cv2
import numpy as np
import mediapipe as mp

class PoseDetector:
    def __init__(self):
        """Initialize MediaPipe pose detection"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame):
        """Process a video frame and return bounding-box-normalized pose landmark coordinates and results."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        normalized_coords = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            
            # Bounding box normalization (better approach)
            min_vals = coords.min(axis=0)
            max_vals = coords.max(axis=0)
            center = (min_vals + max_vals) / 2
            size = (max_vals - min_vals).max()  # Largest dimension for uniform scaling
            
            if size != 0:
                normalized_coords = (coords - center) / size
            else:
                normalized_coords = coords - center
                
        return normalized_coords, results

    def getpoints(self, image):
        """
        Extract normalized pose coordinates from image using MediaPipe
        Returns: Normalized 3D coordinates (33 points * 3 coordinates)
        """
        normalized_coords, results = self.process_frame(image)
        
        if normalized_coords is None:
            # Return zeros if no pose detected
            return np.zeros(99)  # 33 points * 3 coordinates
        
        return normalized_coords.flatten()

    def getpoints_vis(self, image):
        """
        Extract keypoints and return visualization
        Returns: (keypoints, annotated_image)
        """
        normalized_coords, results = self.process_frame(image)
        
        # Create visualization
        annotated_image = image.copy()
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Return flattened coordinates for compatibility
        keypoints = normalized_coords.flatten() if normalized_coords is not None else np.zeros(99)
        return keypoints, annotated_image

    def cosine_similarity_pose(self, pose1, pose2):
        """Calculate cosine similarity between two poses"""
        vec1 = pose1.flatten()
        vec2 = pose2.flatten()
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vec1 = vec1 / norm1
        vec2 = vec2 / norm2
        return np.dot(vec1, vec2)

    def bounding_box(self, coords):
        """
        Calculate bounding box from coordinates (legacy method for compatibility)
        """
        if coords is None or len(coords) == 0:
            return [(0, 0), (0, 0), (0, 0), (0, 0)]
        
        # Convert to 2D if needed
        if len(coords.shape) == 1:
            coords = coords.reshape(-1, 3)
        
        # Use only x,y coordinates for 2D bounding box
        coords_2d = coords[:, :2]
        
        min_x = coords_2d[:, 0].min()
        min_y = coords_2d[:, 1].min()
        max_x = coords_2d[:, 0].max()
        max_y = coords_2d[:, 1].max()
        
        return [(int(min_x), int(min_y)), (int(max_x), int(min_y)),
                (int(max_x), int(max_y)), (int(min_x), int(max_y))]

    def get_new_coords(self, coords, roi_coords):
        """
        Normalize coordinates based on ROI (legacy method for compatibility)
        """
        if len(roi_coords) < 4:
            return coords
        
        # Extract ROI dimensions
        x1, y1 = roi_coords[0]
        x2, y2 = roi_coords[2]
        
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            return coords
        
        # Normalize coordinates
        normalized_coords = coords.copy()
        for i in range(len(coords)):
            normalized_coords[i][0] = (coords[i][0] - y1) / height
            normalized_coords[i][1] = (coords[i][1] - x1) / width
        
        return normalized_coords

    def roi(self, imagepoints):
        """
        Extract ROI and normalize coordinates (legacy method for compatibility)
        """
        # This is a legacy method - the new approach uses bounding box normalization
        # in process_frame() which is much better
        return imagepoints











