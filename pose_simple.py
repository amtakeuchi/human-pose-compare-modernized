import cv2
import numpy as np

class PoseSimple:
    def __init__(self):
        # Use simple detection approach without requiring model files
        self.use_simple_detection = True
        
    def getpoints(self, image_input):
        """
        Extract pose keypoints from image using simple detection
        Returns: list of coordinates [x1, y1, x2, y2, ..., x17, y17, s1, s2, ..., s17, total_score]
        """
        pos_temp_data = []
        total_score = 0
        
        # Simple pose detection using contour detection
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        
        # Detect human-like contours
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely to be a person)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Create 17 keypoints based on the bounding box
            keypoints = []
            for i in range(17):
                # Distribute keypoints across the bounding box
                if i < 5:  # Head and shoulders
                    kp_x = x + w * (0.2 + 0.6 * i / 4)
                    kp_y = y + h * 0.1
                elif i < 10:  # Arms
                    kp_x = x + w * (0.1 + 0.8 * (i - 5) / 4)
                    kp_y = y + h * (0.3 + 0.2 * (i - 5) / 4)
                elif i < 15:  # Body and legs
                    kp_x = x + w * (0.3 + 0.4 * (i - 10) / 4)
                    kp_y = y + h * (0.5 + 0.4 * (i - 10) / 4)
                else:  # Feet
                    kp_x = x + w * (0.2 + 0.6 * (i - 15))
                    kp_y = y + h * 0.9
                
                keypoints.append((kp_x, kp_y))
            
            # Add coordinates
            for kp in keypoints:
                pos_temp_data.append(kp[1])  # y coordinate
                pos_temp_data.append(kp[0])  # x coordinate
            
            # Add confidence scores (simplified)
            for _ in range(17):
                score = 0.8  # Default confidence
                pos_temp_data.append(score)
                total_score += score
                
            pos_temp_data.append(total_score)
        else:
            # If no contours found, return zeros
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
        pos_temp_data = self.getpoints(image_input)
        
        # Create visualization
        black_image = np.zeros((image_input.shape[0], image_input.shape[1], 3), dtype='uint8')
        
        # Draw keypoints
        for i in range(0, 34, 2):
            x = int(pos_temp_data[i + 1])
            y = int(pos_temp_data[i])
            if x > 0 and y > 0:
                cv2.circle(black_image, (x, y), 3, (0, 255, 0), -1)
        
        return pos_temp_data, black_image
    
    def bounding_box(self, coords):
        """Calculate bounding box from coordinates"""
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        # Handle different input formats
        if isinstance(coords, list):
            # If coords is a list of [x, y] pairs
            for coord in coords:
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