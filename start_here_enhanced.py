# USAGE : python3 start_here_enhanced.py --activity "punch - side" --video "test.mp4" --lookup "lookup_enhanced.pickle"

import argparse
from calculations_enhanced import get_Score


ap = argparse.ArgumentParser()
ap.add_argument("-a", "--activity", required=True,
    help="activity to be scored")
ap.add_argument("-v", "--video", required=True,
    help="video file to be scored against")
ap.add_argument("-l", "--lookup", default="lookup_enhanced.pickle",
    help="The pickle file containing the lookup table")
args = vars(ap.parse_args())


g = get_Score(args["lookup"])

try:
    final_score, score_list = g.calculate_Score(args["video"], args["activity"])
    print("Total Score : ", final_score)
    print("Score List : ", score_list)
except ValueError as e:
    print(f"Error: {e}")
    print("Available actions in lookup:")
    for action in g.b.keys():
        print(f"  - {action}")
except Exception as e:
    print(f"Error processing video: {e}") 