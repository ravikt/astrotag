import cv2
import json
import numpy as np
from detector import create_tag_dict

signature = []
world_loc = []
for i in range(20):
    marker_path = 'marker/lasrtag_dictionary/marker_{}.png'.format(i)

    marker = cv2.imread(marker_path)

    dict_sig, dict_world_loc = create_tag_dict(marker, 700)

    signature.append(dict_sig)
    world_loc.append(dict_world_loc)

index = np.array(list(range(0, 20)))
world_loc =np.array(world_loc)
signature = np.array(signature)    
data = {
    "index": index.tolist(),
    "dict_sig": signature.tolist(),
    "dict_world_loc": world_loc.tolist()
}
with open('marker_dictionary.json', 'w') as marker_dict:
    json.dump(data, marker_dict, indent=1)