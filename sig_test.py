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

def analyze_signature(sig_str: str, reference_msgs: tuple, grs_encoder):
    """Detailed analysis of signature decoding"""
    # Convert to int list for GRS decoder
    encoded_grs_int_list = binary_string_to_int_list(sig_str)
    
    # Calculate syndromes before decoding
    syndromes = grs_encoder.calc_syndromes(encoded_grs_int_list)
    print(f"Syndromes: {syndromes[:5]}...")  # Show first 5
    
    # Try decoding
    decoded_grs = grs_encoder.decode(encoded_grs_int_list)
    decoded_msg = int_list_to_binary_string(decoded_grs)[:24]
    
    # Compare with all rotations
    print("\nHamming distances to reference messages:")
    for ref in reference_msgs:
        dist = sum(a != b for a, b in zip(decoded_msg, ref))
        print(f"{ref}: {dist} bits different")
    
    return decoded_msg

def test_signatures():
    sig_folder = 'sig_test'
    reference_msgs = msg
    
    print("GRS Parameters:")
    print(f"Field size: {grs_encoder.q}")
    print(f"Message length (k): {grs_encoder.k}")
    print(f"Codeword length (n): {grs_encoder.n}")
    print(f"Error correction capacity: {(grs_encoder.n - grs_encoder.k)//2}\n")
    
    for img_file in os.listdir(sig_folder):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(sig_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            signature = get_id(img)
            sig_str = ''.join(map(str, signature))
            
            print(f"\nTesting {img_file}:")
            print(f"Raw signature: {sig_str}")
            
            decoded_msg = analyze_signature(sig_str, reference_msgs, grs_encoder)
            print(f"Decoded msg:   {decoded_msg}")
            
            if decoded_msg in reference_msgs:
                print(" Successfully decoded")
            else:
                print(" Decoding failed")
                print("Error pattern analysis:")
                # Show where errors occurred
                ref_msgs_bits = [list(ref) for ref in reference_msgs]
                decoded_bits = list(decoded_msg)
                for i, bit in enumerate(decoded_bits):
                    mismatches = sum(bit != ref[i] for ref in ref_msgs_bits)
                    if mismatches == len(reference_msgs):
                        print(f"Position {i}: All references differ")

if __name__ == "__main__":
    test_signatures()


