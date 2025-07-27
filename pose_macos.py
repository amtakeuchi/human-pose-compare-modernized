import cv2
import numpy as np
import mediapipe as mp

class PoseMacOS:
    def __init__(self):
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
        
    def getpoints(self, image_input):
        """
        Extract pose keypoints from image using MediaPipe
        Returns: list of coordinates [x1, y1, x2, y2, ..., x17, y17, s1, s2, ..., s17, total_score]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        pos_temp_data = []
        total_score = 0
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract coordinates (17 keypoints - MediaPipe has 33, we'll map to 17)
            keypoint_mapping = [
                0,   # nose
                11,  # left shoulder
                12,  # right shoulder
                13,  # left elbow
                14,  # right elbow
                15,  # left wrist
                16,  # right wrist
                23,  # left hip
                24,  # right hip
                25,  # left knee
                26,  # right knee
                27,  # left ankle
                28,  # right ankle
                1,   # left eye
                2,   # right eye
                7,   # left ear
                8    # right ear
            ]
            
            for i in keypoint_mapping:
                if i < len(landmarks):
                    # Normalize coordinates to image dimensions
                    x = landmarks[i].x * image_input.shape[1]
                    y = landmarks[i].y * image_input.shape[0]
                    pos_temp_data.append(y)  # y coordinate
                    pos_temp_data.append(x)  # x coordinate
                else:
                    pos_temp_data.append(0.0)  # y coordinate
                    pos_temp_data.append(0.0)  # x coordinate
            
            # Extract confidence scores
            for i in keypoint_mapping:
                if i < len(landmarks):
                    score = landmarks[i].visibility
                    pos_temp_data.append(score)
                    total_score += score
                else:
                    pos_temp_data.append(0.0)
                    total_score += 0.0
                    
            pos_temp_data.append(total_score)
        else:
            # If no pose detected, return zeros
            for _ in range(17):
                pos_temp_data.append(0.0)  # y coordinates
                pos_temp_data.append(0.0)  # x coordinates
            for _ in range(17):
                pos_temp_data.append(0.0)  # scores
            pos_temp_data.append(0.0)  # total score
            
        return pos_temp_data
    
    def getpoints_vis(self, image_input):
        """
        Extract pose keypoints and return visualization image
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        pos_temp_data = []
        total_score = 0
        
        # Create visualization
        black_image = np.zeros((image_input.shape[0], image_input.shape[1], 3), dtype='uint8')
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                black_image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            
            landmarks = results.pose_landmarks.landmark
            
            # Extract coordinates (17 keypoints)
            keypoint_mapping = [
                0,   # nose
                11,  # left shoulder
                12,  # right shoulder
                13,  # left elbow
                14,  # right elbow
                15,  # left wrist
                16,  # right wrist
                23,  # left hip
                24,  # right hip
                25,  # left knee
                26,  # right knee
                27,  # left ankle
                28,  # right ankle
                1,   # left eye
                2,   # right eye
                7,   # left ear
                8    # right ear
            ]
            
            for i in keypoint_mapping:
                if i < len(landmarks):
                    x = landmarks[i].x * image_input.shape[1]
                    y = landmarks[i].y * image_input.shape[0]
                    pos_temp_data.append(y)
                    pos_temp_data.append(x)
                else:
                    pos_temp_data.append(0.0)
                    pos_temp_data.append(0.0)
            
            # Extract confidence scores
            for i in keypoint_mapping:
                if i < len(landmarks):
                    score = landmarks[i].visibility
                    pos_temp_data.append(score)
                    total_score += score
                else:
                    pos_temp_data.append(0.0)
                    total_score += 0.0
                    
            pos_temp_data.append(total_score)
        else:
            # If no pose detected, return zeros
            for _ in range(17):
                pos_temp_data.append(0.0)
                pos_temp_data.append(0.0)
            for _ in range(17):
                pos_temp_data.append(0.0)
            pos_temp_data.append(0.0)
            
        return pos_temp_data, black_image
    
    def bounding_box(self, coords):
        """Calculate bounding box from coordinates"""
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for i in range(0, len(coords), 2):
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
    
    def roi(self, imagepoints):
        """Extract ROI coordinates"""
        coords_new_reshaped = imagepoints[0:34]
        coords_new = np.asarray(coords_new_reshaped).reshape(17, 2)
        
        # Convert to list format for bounding_box
        coords_list = coords_new.tolist()
        roi_coords = self.bounding_box(coords_list)
        coords_new = self.get_new_coords(coords_new, roi_coords)
        coords_new = coords_new.reshape(34,)
        coords_new = np.concatenate((coords_new[0:34], imagepoints[34:52]))
        return coords_new
    
    def get_new_coords(self, coords, fun_bound):
        """Normalize coordinates relative to bounding box"""
        coords[:, :1] = coords[:, :1] - fun_bound[0][0]
        coords[:, 1:2] = coords[:, 1:2] - fun_bound[0][1]
        return coords 