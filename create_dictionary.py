import cv2
import numpy as np
from detector import create_tag_dict
from read_sig import get_keypoints, get_id
# from grs import Generalized_Reed_Solomon, binary_string_to_int_list, int_list_to_binary_string
from grs import FileHandler
from custom_descriptor.fk_desc import fk_combined


num_markers = 20
signature = []
world_loc = []

# grs_encoder = Generalized_Reed_Solomon(2, 48, 24, 1, 1, None, False, False)

data = {}
keypoints = get_keypoints('keypoints.txt')


for i in range(num_markers):
    marker_path = f'marker/marker_{i}.png'

    # Read and rotate markers
    marker = cv2.imread(marker_path, cv2.IMREAD_GRAYSCALE)
    marker_90 = cv2.rotate(marker, cv2.ROTATE_90_CLOCKWISE)
    marker_180 = cv2.rotate(marker, cv2.ROTATE_180)
    marker_270 = cv2.rotate(marker, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Define world points for each orientation
    world_points = {
        "0": np.float32([[0,0,0], [30,0,0], [30,30,0], [0,30,0]]),
        "90": np.float32([[0,30,0], [0,0,0], [30,0,0], [30,30,0]]),
        "180": np.float32([[30,30,0], [0,30,0], [0,0,0], [30,0,0]]),
        "270": np.float32([[30,0,0], [30,30,0], [0,30,0], [0,0,0]])
    }

    # Get signatures for each rotation
    # signatures = {
    #     "0": ''.join(map(str, get_id(marker))),
    #     "90": ''.join(map(str, get_id(marker_90))),
    #     "180": ''.join(map(str, get_id(marker_180))),
    #     "270": ''.join(map(str, get_id(marker_270)))
    # }


    # Get signatures for each rotation using Fixed Keypoint Descriptor
    signatures = {
        "0": ''.join(fk_combined(marker, keypoints).astype(str).flatten()),
        "90": ''.join(fk_combined(marker_90, keypoints).astype(str).flatten()),
        "180": ''.join(fk_combined(marker_180, keypoints).astype(str).flatten()),
        "270": ''.join(fk_combined(marker_270, keypoints).astype(str).flatten())
    }

    # Store both signature and world points
    data[str(i)] = {
        angle: {
            "signature": signatures[angle],
            "world_points": world_points[angle].tolist()
        }
        for angle in ["0", "90", "180", "270"]
    }

# Save to JSON
output_file = 'fk_dictionary.json'
FileHandler.save_to_json(data, output_file)
print(f"Dictionary saved to {output_file}")