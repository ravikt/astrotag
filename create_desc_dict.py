import cv2
import numpy as np
import mfd_descriptor
import json
from read_sig import get_keypoints


num_markers = 3
signature = []
world_loc = []
keypoints = get_keypoints('keypoints.txt')

for i in range(num_markers):
    marker_path = 'marker/lasrtag_dictionary/thesis_b_marker_{}.png'.format(i)
    marker = cv2.imread(marker_path, cv2.IMREAD_GRAYSCALE)

    descriptors = mfd_descriptor.mfd_combined_descriptor(marker, keypoints)

    signature.append(descriptors.tolist())
    world_loc.append([1, 2, 3]) # Placeholder for world location 


index = np.array(list(range(0, num_markers)))
world_loc =np.array(world_loc)
signature = np.array(signature)    
data = {
    "index": index.tolist(),
    "dict_sig": signature.tolist(),
    "dict_world_loc": world_loc.tolist()
}
with open('lasrtag_dictionary.json', 'w') as marker_dict:
    json.dump(data, marker_dict, indent=1)
