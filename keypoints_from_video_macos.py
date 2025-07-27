import cv2
import time
import math
import numpy 
import numpy as np
from pose_macos import PoseMacOS
from score import Score
import pickle
import argparse

# USAGE : python3 keypoints_from_video_macos.py --activity "punch - side" --video "test.mp4" 

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--activity", required=True,
    help="activity to be recorder")
ap.add_argument("-v", "--video", required=True,
    help="video file from which keypoints are to be extracted")
ap.add_argument("-l", "--lookup", default="lookup_new.pickle",
    help="The pickle file to dump the lookup table")
args = vars(ap.parse_args())


def main():
    a = PoseMacOS()
    b = []
    c = {}
    
    cap = cv2.VideoCapture(args["video"])
    i = 1

    if not cap.isOpened():
        print("Error in opening video")
        return
        
    while cap.isOpened():
        ret_val, image = cap.read()
        if ret_val:
            image = cv2.resize(image, (372, 495))
            input_points, input_black_image = a.getpoints_vis(image)
            input_points = input_points[0:34]
            print(f"Frame {i}: {len(input_points)} points extracted")
            input_new_coords = a.roi(input_points)
            input_new_coords = input_new_coords[0:34]
            input_new_coords = np.asarray(input_new_coords).reshape(17, 2)
            b.append(input_new_coords)
            cv2.imshow("Pose Detection (macOS)", input_black_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i = i + 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    b = np.array(b)
    
    print(f"Extracted {b.shape[0]} frames")
    print(f"Shape: {b.shape}")
    print("Lookup Table Created")
    c[args["activity"]] = b
    f = open(args["lookup"], 'wb')
    pickle.dump(c, f)
    f.close()
    print(f"Saved to {args['lookup']}")


if __name__ == "__main__":
    main() 