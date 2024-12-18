import numpy as np
import cv2 
import matplotlib.pyplot as plt
from read_sig import get_id
from grs import Generalized_Reed_Solomon, binary_string_to_int_list, int_list_to_binary_string,pad_encoded_message
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
    """Analyze signature decoding process"""
    encoded_grs_int_list = binary_string_to_int_list(sig_str)
    
    # Direct decoding without syndrome calculation
    try:
        decoded_grs = grs_encoder.decode(encoded_grs_int_list)
        decoded_msg = int_list_to_binary_string(decoded_grs)[:24]
        
        print("\nHamming distances to reference messages:")
        for ref in reference_msgs:
            dist = sum(a != b for a, b in zip(decoded_msg, ref))
            print(f"{ref}: {dist} bits different")
        
        return decoded_msg
    except Exception as e:
        print(f"Decoding error: {e}")
        return None

def analyze_270_variants():
    # Load all 270° variants
    base = cv2.imread('sig_test/ref_270.png', cv2.IMREAD_GRAYSCALE)
    var1 = cv2.imread('sig_test/ref_270_1.png', cv2.IMREAD_GRAYSCALE) 
    var2 = cv2.imread('sig_test/ref_270_2.png', cv2.IMREAD_GRAYSCALE)

    # Get signatures
    sig_base = ''.join(map(str, get_id(base)))
    sig_var1 = ''.join(map(str, get_id(var1)))
    sig_var2 = ''.join(map(str, get_id(var2)))

    print("Reference 270° signature:")
    print(sig_base)
    
    # Compare variants
    print("\nVariant 1 differences:")
    diffs1 = [(i, sig_var1[i]) for i in range(48) if sig_var1[i] != sig_base[i]]
    print(f"Total differences: {len(diffs1)}")
    print(f"Different positions: {[pos for pos,_ in diffs1]}")
    
    print("\nVariant 2 differences:") 
    diffs2 = [(i, sig_var2[i]) for i in range(48) if sig_var2[i] != sig_base[i]]
    print(f"Total differences: {len(diffs2)}")
    print(f"Different positions: {[pos for pos,_ in diffs2]}")

    # Try decoding each
    print("\nAttempting decodes:")
    for sig, name in [(sig_base, "base"), (sig_var1, "var1"), (sig_var2, "var2")]:
        int_list = binary_string_to_int_list(sig)
        decoded = grs_encoder.decode(int_list)
        msg = int_list_to_binary_string(decoded)[:24]
        print(f"\n{name}:")
        print(f"Decoded: {msg}")
        print(f"Success: {msg in msg}")

def test_signatures():
    # Add variant analysis
    analyze_270_variants()
    # Rest of original test code...
    sig_folder = 'sig_test'
    reference_msgs = msg
    
    print("GRS Parameters:")
    print(f"Field size: 2")
    print(f"Message length: 24")
    print(f"Codeword length: 48")
    print(f"Error correction capacity: 12\n")
    
    for img_file in os.listdir(sig_folder):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(sig_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            signature = get_id(img)
            sig_str = ''.join(map(str, signature))
            
            print(f"\nTesting {img_file}:")
            print(f"Raw signature: {sig_str}")
            
            decoded_msg = analyze_signature(sig_str, reference_msgs, grs_encoder)
            if decoded_msg:
                print(f"Decoded msg: {decoded_msg}")
                if decoded_msg in reference_msgs:
                    print("Successfully decoded")
                else:
                    print("Decoding failed")
            else:
                print("Decoding failed with error")

def test_grs_error_correction():
    """Test GRS error correction with known error pattern"""
    # Base reference from 270° image
    base = "011011100011011110110100111110010001010100101111"
    
    # Create test pattern with known errors
    test = list(base)
    error_positions = [3, 29, 30]
    for pos in error_positions:
        test[pos] = '1' if test[pos] == '0' else '0'
    test = ''.join(test)
    
    print("Test Pattern Analysis:")
    print(f"Base:     {base}")
    print(f"Modified: {test}")
    print(f"Errors at positions: {error_positions}")
    
    # Try encoding/decoding cycle
    try:
        # First encode known message
        msg = base[:24]
        msg_ints = binary_string_to_int_list(msg)
        encoded = grs_encoder.encode(msg_ints)
        encoded_str = int_list_to_binary_string(encoded)
        print(f"\nEncoding check:")
        print(f"Original msg: {msg}")
        print(f"Encoded:      {encoded_str}")
        
        # Then try decoding with errors
        test_ints = binary_string_to_int_list(test)
        decoded = grs_encoder.decode(test_ints)
        decoded_str = int_list_to_binary_string(decoded)[:24]
        print(f"\nDecoding result:")
        print(f"Decoded msg:  {decoded_str}")
        print(f"Success: {decoded_str == msg}")
        
    except Exception as e:
        print(f"Error: {e}")

def test_grs_codec():
    """Verify GRS encoder/decoder functionality"""
    # Test with known 270° message
    msg = "011011100011011110110100"  # 270° message
    
    print("GRS Codec Test:")
    print(f"Original message: {msg}")
    
    # Test encoding
    msg_ints = binary_string_to_int_list(msg)
    encoded = grs_encoder.encode(msg_ints)
    encoded_str = int_list_to_binary_string(encoded)
    print(f"Encoded codeword: {encoded_str}")
    
    # Test decoding without errors
    decoded = grs_encoder.decode(encoded)
    decoded_str = int_list_to_binary_string(decoded)[:24]
    print(f"Decoded message: {decoded_str}")
    print(f"Round-trip success: {msg == decoded_str}")
    
    # Test with known error pattern
    error_pos = [3, 29, 30]
    corrupted = list(encoded_str)
    for pos in error_pos:
        corrupted[pos] = '1' if corrupted[pos] == '0' else '0'
    corrupted = ''.join(corrupted)
    print(f"\nCorrupted codeword: {corrupted}")
    
    # Try decoding corrupted message
    corrupted_ints = binary_string_to_int_list(corrupted)
    try:
        decoded = grs_encoder.decode(corrupted_ints)
        decoded_str = int_list_to_binary_string(decoded)[:24]
        print(f"Decoded from corruption: {decoded_str}")
        print(f"Error correction success: {msg == decoded_str}")
    except Exception as e:
        print(f"Decoding failed: {e}")

def test_error_patterns():
    """Test GRS decoder with different error patterns"""
    # msg = "011011100011011110110100"
    msg = "100101101101011001101110"
    msg_ints = binary_string_to_int_list(msg)
    
    # Get clean codeword
    codeword = grs_encoder.encode(msg_ints)
    codeword = pad_encoded_message(codeword, 48)
    codeword_str = int_list_to_binary_string(codeword)
    print("Original codeword: " + codeword_str)
    # Test patterns:
    patterns = [
        ([0], "Start errors"),
        ([0,1,2], "Start errors"),
        ([22,23,24], "Message-parity boundary"),
        ([45,46,47], "End errors"),
        ([38,39,40,41,42,43,44,45,46,47], "End errors"),
        ([3,29,30], "Original failing pattern")
    ]
    
    for positions, desc in patterns:
        print(f"\nTesting {desc}:")
        # Introduce errors
        corrupted = list(codeword_str)
        for pos in positions:
            corrupted[pos] = '1' if corrupted[pos] == '0' else '0'
        corrupted = ''.join(corrupted)
        
        # Try decoding
        try:
            encoded_grs_int_list=binary_string_to_int_list(corrupted)
            decoded_grs = grs_encoder.decode(encoded_grs_int_list)
            decoded_msg = int_list_to_binary_string(decoded_grs)[:24]
            print(f"Error positions: {positions}")
            print(f"Success: {decoded_msg == msg}")
        except Exception as e:
            print(f"Decoding failed: {e}")

def test_systematic_grs():
    """Test systematic GRS properties"""
    msg = "100101101101011001101110"
    msg_ints = binary_string_to_int_list(msg)
    
    # Get systematic codeword
    codeword = grs_encoder.encode(msg_ints)
    codeword = pad_encoded_message(codeword, 48)
    codeword_str = int_list_to_binary_string(codeword)
    
    # Verify systematic property
    print("Systematic Code Check:")
    print(f"Message:         {msg}")
    print(f"Codeword[0:24]: {codeword_str[:24]}")
    print(f"Parity[24:48]:  {codeword_str[24:]}")
    print(f"Is systematic:   {msg == codeword_str[:24]}\n")
    
    # Test error patterns aligned with systematic structure
    patterns = [
        # Message region only
        ([0,1,2], "Message only errors"),
        # Parity region only 
        ([38,39,40,41,42,43,44,45,46,47], "Parity only errors"),
        # Cross boundary
        ([22,23,24,25], "Boundary errors"),
        # Original failing pattern
        ([3,29,30], "Cross region errors")
    ]
    
    for positions, desc in patterns:
        print(f"\nTesting {desc}:")
        corrupted = list(codeword_str)
        for pos in positions:
            corrupted[pos] = '1' if corrupted[pos] == '0' else '0'
        corrupted = ''.join(corrupted)
        
        # Show corruption pattern
        print(f"Message region:  {''.join(['E' if i in positions else '_' for i in range(24)])}")
        print(f"Parity region:   {''.join(['E' if i in positions else '_' for i in range(24,48)])}")
        
        try:
            decoded = grs_encoder.decode(binary_string_to_int_list(corrupted))
            decoded_msg = int_list_to_binary_string(decoded)[:24]
            print(f"Success: {decoded_msg == msg}")
        except Exception as e:
            print(f"Failed: {e}")

def test_parity_error_limits():
    """Test error correction limits in parity region"""
    msg = "100101101101011001101110"
    msg_ints = binary_string_to_int_list(msg)
    
    # Get systematic codeword
    codeword = grs_encoder.encode(msg_ints)
    codeword = pad_encoded_message(codeword, 48)
    codeword_str = int_list_to_binary_string(codeword)
    
    # Test increasing error counts in parity region
    for num_errors in range(1, 13):  # Up to correction capacity
        positions = list(range(48-num_errors, 48))  # Take errors from end
        
        print(f"\nTesting {num_errors} parity region errors:")
        corrupted = list(codeword_str)
        for pos in positions:
            corrupted[pos] = '1' if corrupted[pos] == '0' else '0'
        corrupted = ''.join(corrupted)
        
        # Show error pattern
        print(f"Error positions: {positions}")
        
        try:
            decoded = grs_encoder.decode(binary_string_to_int_list(corrupted))
            decoded_msg = int_list_to_binary_string(decoded)[:24]
            print(f"Success: {decoded_msg == msg}")
        except Exception as e:
            print(f"Failed: {e}")

def test_parity_pattern():
    """Test even/odd error patterns in parity region"""
    msg = "100101101101011001101110"
    msg_ints = binary_string_to_int_list(msg)
    
    # Get systematic codeword
    codeword = grs_encoder.encode(msg_ints)
    codeword = pad_encoded_message(codeword, 48)
    codeword_str = int_list_to_binary_string(codeword)
    
    print("Systematic codeword analysis:")
    print(f"Message part: {codeword_str[:24]}")
    print(f"Parity part:  {codeword_str[24:]}\n")
    
    # Test even vs odd patterns
    patterns = [
        ([47], "Single error"),
        ([46,47], "Double error"),
        ([44,45,46,47], "Four consecutive"),
        ([44,46], "Two spaced"),
        ([40,42,44,46], "Four spaced"),
        ([38,40,42,44,46], "Five spaced")
    ]
    
    for positions, desc in patterns:
        print(f"\nTesting {desc}:")
        corrupted = list(codeword_str)
        for pos in positions:
            corrupted[pos] = '1' if corrupted[pos] == '0' else '0'
        corrupted = ''.join(corrupted)
        
        # Show parity bits affected
        parity_pattern = ['_'] * 24
        for p in positions:
            parity_pattern[p-24] = 'E'
        print(f"Parity pattern: {''.join(parity_pattern)}")
        
        try:
            decoded = grs_encoder.decode(binary_string_to_int_list(corrupted))
            decoded_msg = int_list_to_binary_string(decoded)[:24]
            print(f"Success: {decoded_msg == msg}")
        except Exception as e:
            print(f"Failed: {e}")

def test_error_spacing():
    """Test impact of error spacing in parity region"""
    msg = "100101101101011001101110"
    msg_ints = binary_string_to_int_list(msg)
    
    codeword = grs_encoder.encode(msg_ints)
    codeword = pad_encoded_message(codeword, 48)
    codeword_str = int_list_to_binary_string(codeword)
    
    # Test patterns with consistent spacing
    patterns = [
        # Even counts, different spacings
        ([46,47], "Adjacent pair"),
        ([44,47], "Spaced pair"),
        ([44,45,46,47], "Adjacent quad"),
        ([42,44,46,47], "Mixed spacing quad"),
        ([40,42,44,46], "Even spaced quad"),
        
        # Even count, different regions
        ([23,24], "Message-parity boundary"),
        ([0,47], "Start-end pair"),
        ([24,47], "Parity boundary-end"),
        
        # Control group
        ([47], "Single end"),
        ([46,47,48], "Triple end")
    ]
    
    print("Testing error spacing patterns:")
    for positions, desc in patterns:
        print(f"\n{desc}:")
        corrupted = list(codeword_str)
        for pos in positions:
            if pos < len(corrupted):
                corrupted[pos] = '1' if corrupted[pos] == '0' else '0'
        corrupted = ''.join(corrupted)
        
        # Visualize full error pattern
        pattern = ['_'] * 48
        for p in positions:
            if p < 48:
                pattern[p] = 'E'
        print(f"Full pattern: {''.join(pattern[:24])}|{''.join(pattern[24:])}")
        
        try:
            decoded = grs_encoder.decode(binary_string_to_int_list(corrupted))
            decoded_msg = int_list_to_binary_string(decoded)[:24]
            print(f"Success: {decoded_msg == msg}")
        except Exception as e:
            print(f"Failed: {e}")

def test_error_position_boundaries():
    """Test error correction near region boundaries"""
    msg = "100101101101011001101110"
    msg_ints = binary_string_to_int_list(msg)
    
    codeword = grs_encoder.encode(msg_ints)
    codeword = pad_encoded_message(codeword, 48)
    codeword_str = int_list_to_binary_string(codeword)
    
    # Test boundary patterns
    patterns = [
        # Parity region only
        ([24,25], "Start of parity"),
        ([46,47], "End of parity"),
        ([24,47], "Parity bounds"),
        
        # Cross boundary
        ([23,24], "Message-parity"),
        ([22,23,24,25], "Boundary span"),
        
        # Control
        ([24,26,28,30], "Even spaced parity")
    ]
    
    for positions, desc in patterns:
        print(f"\n{desc}:")
        corrupted = list(codeword_str)
        for pos in positions:
            corrupted[pos] = '1' if corrupted[pos] == '0' else '0'
        
        print(f"Pattern: {''.join(['E' if i in positions else '_' for i in range(48)])}")
        
        try:
            decoded = grs_encoder.decode(binary_string_to_int_list(''.join(corrupted)))
            print(f"Success: {int_list_to_binary_string(decoded)[:24] == msg}")
        except Exception as e:
            print(f"Failed: {e}")

def test_systematic_properties():
    """Test systematic GRS code properties"""
    msg = "100101101101011001101110"
    msg_ints = binary_string_to_int_list(msg)
    
    codeword = grs_encoder.encode(msg_ints)
    codeword = pad_encoded_message(codeword, 48)
    codeword_str = int_list_to_binary_string(codeword)
    
    # Fix: Use test_cases instead of patterns
    test_cases = [
        # Test parity region even/odd
        ([44,46], "Even parity pair"),
        ([44,45,46], "Odd parity triple"),
        ([40,41,42,43], "Consecutive block"),
        ([40,42,44,46], "Spaced block"),
        ([23,24,25], "Boundary errors"),
        ([24,25,26], "Parity start")
    ]
    
    print("Testing systematic GRS properties:")
    for positions, desc in test_cases:  # Fixed: Use test_cases instead of patterns
        print(f"\n{desc}:")
        corrupted = list(codeword_str)
        for pos in positions:
            corrupted[pos] = '1' if corrupted[pos] == '0' else '0'
        
        msg_errors = sum(1 for p in positions if p < 24)
        par_errors = sum(1 for p in positions if p >= 24)
        print(f"Message errors: {msg_errors}, Parity errors: {par_errors}")
        print(f"Pattern: {''.join(['E' if i in positions else '_' for i in range(48)])}")
        
        try:
            decoded = grs_encoder.decode(binary_string_to_int_list(''.join(corrupted)))
            print(f"Success: {int_list_to_binary_string(decoded)[:24] == msg}")
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    test_error_patterns()


