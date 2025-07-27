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

    def getpoints(self, image):
        """
        Extract 17 keypoints from image using MediaPipe
        Returns: 34-element array (17 points * 2 coordinates)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            # Return zeros if no pose detected
            return np.zeros(34)
        
        # Extract 17 keypoints (mapping MediaPipe's 33 points to 17)
        keypoints = []
        
        # MediaPipe pose landmarks mapping to 17 keypoints
        landmark_mapping = [
            self.mp_pose.PoseLandmark.NOSE,           # 0
            self.mp_pose.PoseLandmark.LEFT_EYE,       # 1
            self.mp_pose.PoseLandmark.RIGHT_EYE,      # 2
            self.mp_pose.PoseLandmark.LEFT_EAR,       # 3
            self.mp_pose.PoseLandmark.RIGHT_EAR,      # 4
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,  # 5
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER, # 6
            self.mp_pose.PoseLandmark.LEFT_ELBOW,     # 7
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,    # 8
            self.mp_pose.PoseLandmark.LEFT_WRIST,     # 9
            self.mp_pose.PoseLandmark.RIGHT_WRIST,    # 10
            self.mp_pose.PoseLandmark.LEFT_HIP,       # 11
            self.mp_pose.PoseLandmark.RIGHT_HIP,      # 12
            self.mp_pose.PoseLandmark.LEFT_KNEE,      # 13
            self.mp_pose.PoseLandmark.RIGHT_KNEE,     # 14
            self.mp_pose.PoseLandmark.LEFT_ANKLE,     # 15
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,    # 16
        ]
        
        for landmark_id in landmark_mapping:
            landmark = results.pose_landmarks.landmark[landmark_id]
            # Convert normalized coordinates to pixel coordinates
            h, w, _ = image.shape
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            keypoints.extend([y, x])  # y, x format to match original
        
        return np.array(keypoints)

    def getpoints_vis(self, image):
        """
        Extract keypoints and return visualization
        Returns: (keypoints, annotated_image)
        """
        keypoints = self.getpoints(image)
        
        # Create visualization
        annotated_image = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return keypoints, annotated_image

    def bounding_box(self, coords):
        """
        Calculate bounding box from coordinates
        Args:
            coords: List of [x, y] coordinate pairs
        Returns:
            List of 4 corner points [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
        """
        if not coords:
            return [(0, 0), (0, 0), (0, 0), (0, 0)]
        
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        # Handle different input formats
        if isinstance(coords, list):
            # If coords is a list of [x, y] pairs
            for coord in coords:
                if len(coord) >= 2:
                    x, y = coord[0], coord[1]
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
        else:
            # If coords is a flat list [y1, x1, y2, x2, ...]
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x, y = coords[i+1], coords[i]  # x, y coordinates
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
        
        return [(int(min_x), int(min_y)), (int(max_x), int(min_y)),
                (int(max_x), int(max_y)), (int(min_x), int(max_y))]

    def get_new_coords(self, coords, roi_coords):
        """
        Normalize coordinates based on ROI
        Args:
            coords: 17x2 array of keypoints
            roi_coords: ROI bounding box coordinates
        Returns:
            Normalized coordinates
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
        Extract ROI and normalize coordinates
        Args:
            imagepoints: 52-element array (34 keypoints + 18 additional features)
        Returns:
            Normalized coordinates
        """
        coords_new_reshaped = imagepoints[0:34]
        coords_new = np.asarray(coords_new_reshaped).reshape(17, 2)
        
        # Convert to list format for bounding_box
        coords_list = coords_new.tolist()
        roi_coords = self.bounding_box(coords_list)
        
        coords_new = self.get_new_coords(coords_new, roi_coords)
        coords_new = coords_new.reshape(34,)
        coords_new = np.concatenate((coords_new[0:34], imagepoints[34:52]))
        
        return coords_new











