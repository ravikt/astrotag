import cv2
import numpy as np
from detector import create_tag_dict
from read_sig import get_id
from grs import Generalized_Reed_Solomon, binary_string_to_int_list, int_list_to_binary_string
from grs import FileHandler



num_markers = 20
signature = []
world_loc = []

grs_encoder = Generalized_Reed_Solomon(2, 48, 24, 1, 1, None, False, False)

data = {}


def decode_msg(encoded_grs_bits):
    encoded_grs_int_list = binary_string_to_int_list(encoded_grs_bits)
    decoded_grs = grs_encoder.decode(encoded_grs_int_list)
    decoded_grs_bits = int_list_to_binary_string(decoded_grs)
    message = decoded_grs_bits[:24]

    return message


for i in range(num_markers):
    marker_path = 'marker/marker_{}.png'.format(i)

    marker = cv2.imread(marker_path, cv2.IMREAD_GRAYSCALE)
    marker_90 = cv2.rotate(marker, cv2.ROTATE_90_CLOCKWISE)
    marker_180 = cv2.rotate(marker, cv2.ROTATE_180)
    marker_270 = cv2.rotate(marker, cv2.ROTATE_90_COUNTERCLOCKWISE)

    encoded_grs_bits_0 = ''.join(map(str, get_id(marker)))
    encoded_grs_bits_90 = ''.join(map(str, get_id(marker_90)))
    encoded_grs_bits_180 = ''.join(map(str, get_id(marker_180)))
    encoded_grs_bits_270 = ''.join(map(str, get_id(marker_270)))

    message_0 = decode_msg(encoded_grs_bits_0)
    message_90 = decode_msg(encoded_grs_bits_90)
    message_180 = decode_msg(encoded_grs_bits_180)
    message_270 = decode_msg(encoded_grs_bits_270)


    data[str(i)] = {
            "0": message_0,
            "90": message_90,
            "180": message_180,
            "270": message_270
        }


output_file='signature_dictionary.json'

FileHandler.save_to_json(data, output_file)
print(f"Data saved to {output_file}")