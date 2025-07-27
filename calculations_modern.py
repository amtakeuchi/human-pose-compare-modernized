import pickle
import cv2
import numpy as np
from pose_modern import PoseModern
from score import Score
from dtaidistance import dtw


class get_Score(object):
    def __init__(self, lookup='lookup.pickle'):
        self.a = PoseModern()
        self.s = Score()
        try:
            self.b = pickle.load(open(lookup, 'rb'))
        except FileNotFoundError:
            print(f"Warning: Lookup file {lookup} not found. Creating empty lookup.")
            self.b = {}
        self.input_test = []

    def get_action_coords_from_dict(self, action):
        for (k, v) in self.b.items():
            if k == action:
                model_array = v
                no_of_frames = v.shape[0]
                return model_array, no_of_frames
        raise ValueError(f"Action '{action}' not found in lookup table")

    def calculate_Score(self, video, action):
        model_array, j = self.get_action_coords_from_dict(action)
        cap = cv2.VideoCapture(video)
        i = 0
        
        if not cap.isOpened():
            print("Error in opening video")
            return 0, []
            
        while cap.isOpened():
            ret_val, image = cap.read()
            if ret_val:
                # Resize image to match expected dimensions
                image = cv2.resize(image, (372, 495))
                input_points = self.a.getpoints(image)
                input_new_coords = np.asarray(self.a.roi(input_points)[0:34]).reshape(17, 2)
                self.input_test.append(input_new_coords)
                i = i + 1
            else:
                break
                
        cap.release()
        
        if i == 0:
            print("No frames processed from video")
            return 0, []
            
        final_score, score_list = self.s.compare(np.asarray(self.input_test), np.asarray(model_array), j, i)
        return final_score, score_list 