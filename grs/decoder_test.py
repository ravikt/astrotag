import random
from typing import List, Dict
from rs_codec import ReedSolomonCodec, Generalized_Reed_Solomon
from file_handler import FileHandler
from message_generator import generate_unique_binary_strings, binary_string_to_int_list, int_list_to_binary_string, pad_encoded_message

grs_encoder = Generalized_Reed_Solomon(2, 48, 24, 1, 1, None, False, False)

message = "010010001110001000011011"
        # codeword = "010010001110001000011011110110101001111111001000"

# Verify GRS codeword
# encoded_grs_bits = "010010001110001000011011110110101001000000110111" # last 12 bits are flipped
encoded_grs_bits = "010010001110001000011011110110101001111111001000"

encoded_grs_int_list = binary_string_to_int_list(encoded_grs_bits)
decoded_grs = grs_encoder.decode(encoded_grs_int_list)
decoded_grs_bits = int_list_to_binary_string(decoded_grs)
print(f"Decoded GRS Message: {decoded_grs_bits[:24]}")
if decoded_grs_bits[:24] != message:
    print(f"Verification failed for GRS codeword")
    print(f"Expected: {message}")
    print(f"Decoded:  {decoded_grs_bits[:24]}")
else:
    print("Verification succeeded for GRS codeword")


# from typing import List, Dict
# from rs_codec import ReedSolomonCodec, Generalized_Reed_Solomon
# from file_handler import FileHandler
# from message_generator import generate_unique_binary_strings, binary_string_to_int_list, int_list_to_binary_string, pad_encoded_message

# grs_encoder = Generalized_Reed_Solomon(2, 48, 24, 1, 1, None, False, False)

# message = "010010001110001000011011"
# codeword = "010010001110001000011011110110101001111111001000"

# # Function to introduce errors at specified positions
# def introduce_errors(codeword: str, error_positions: List[int]) -> str:
#     codeword_list = list(codeword)
#     for pos in error_positions:
#         codeword_list[pos] = '1' if codeword_list[pos] == '0' else '0'
#     return ''.join(codeword_list)

# # Test with errors at the beginning
# error_positions = list(range(12))  # Introduce errors in the first 12 bits
# encoded_grs_bits = introduce_errors(codeword, error_positions)

# encoded_grs_int_list = binary_string_to_int_list(encoded_grs_bits)
# decoded_grs = grs_encoder.decode(encoded_grs_int_list)
# decoded_grs_bits = int_list_to_binary_string(decoded_grs)
# print(f"Decoded GRS Message: {decoded_grs_bits[:24]}")
# if decoded_grs_bits[:24] != message:
#     print(f"Verification failed for GRS codeword with errors at the beginning")
#     print(f"Expected: {message}")
#     print(f"Decoded:  {decoded_grs_bits[:24]}")
# else:
#     print("Verification succeeded for GRS codeword with errors at the beginning")

# # Test with errors in the middle
# error_positions = list(range(12, 24))  # Introduce errors in the middle 12 bits
# encoded_grs_bits = introduce_errors(codeword, error_positions)

# encoded_grs_int_list = binary_string_to_int_list(encoded_grs_bits)
# decoded_grs = grs_encoder.decode(encoded_grs_int_list)
# decoded_grs_bits = int_list_to_binary_string(decoded_grs)
# print(f"Decoded GRS Message: {decoded_grs_bits[:24]}")
# if decoded_grs_bits[:24] != message:
#     print(f"Verification failed for GRS codeword with errors in the middle")
#     print(f"Expected: {message}")
#     print(f"Decoded:  {decoded_grs_bits[:24]}")
# else:
#     print("Verification succeeded for GRS codeword with errors in the middle")

# # Test with errors at the end
# error_positions = list(range(36, 48))  # Introduce errors in the last 12 bits
# encoded_grs_bits = introduce_errors(codeword, error_positions)

# encoded_grs_int_list = binary_string_to_int_list(encoded_grs_bits)
# decoded_grs = grs_encoder.decode(encoded_grs_int_list)
# decoded_grs_bits = int_list_to_binary_string(decoded_grs)
# print(f"Decoded GRS Message: {decoded_grs_bits[:24]}")
# if decoded_grs_bits[:24] != message:
#     print(f"Verification failed for GRS codeword with errors at the end")
#     print(f"Expected: {message}")
#     print(f"Decoded:  {decoded_grs_bits[:24]}")
# else:
#     print("Verification succeeded for GRS codeword with errors at the end")