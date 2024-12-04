import numpy as np
import cv2 
import matplotlib.pyplot as plt
from read_sig import get_id
from grs import Generalized_Reed_Solomon, binary_string_to_int_list, int_list_to_binary_string
import os


grs_encoder = Generalized_Reed_Solomon(2, 48, 24, 1, 1, None, False, False)

msg = ("100101101101011001101110",
     "001111011010001111000111",
     "010110001111110100101011",
     "011011100011011110110100")

def decode_grs_signature(sig, grs_encoder):
  """
  Decode a GRS signature using the provided encoder
  
  Args:
    sig: Binary signature to decode
    grs_encoder: Generalized Reed Solomon encoder/decoder instance
  
  Returns:
    str: Decoded binary string (first 24 bits)
  """
  encoded_grs_bits = ''.join(map(str, sig))
  encoded_grs_int_list = binary_string_to_int_list(encoded_grs_bits)
  decoded_grs = grs_encoder.decode(encoded_grs_int_list)
  decoded_grs_bits = int_list_to_binary_string(decoded_grs)
  
  return decoded_grs_bits[:24]


def test_signatures():
    sig_folder = 'sig_test'
    reference_msgs = msg  # Using predefined messages tuple
    
    print("Reference messages:", reference_msgs)
    
    for img_file in os.listdir(sig_folder):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(sig_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Get and decode signature
            signature = get_id(img)
            decoded_msg = decode_grs_signature(signature, grs_encoder)
            
            print(f"\nTesting {img_file}:")
            print(f"Raw signature: {''.join(map(str, signature))}")
            print(f"Decoded msg:   {decoded_msg}")
            
            # Let GRS decoder handle error correction
            if decoded_msg in reference_msgs:
                print(" Successfully decoded to reference message")
            else:
                print(" Decoding failed - message not in reference set")
                print("Reference messages:")
                for ref in reference_msgs:
                    print(f"- {ref}")

if __name__ == "__main__":
    test_signatures()


