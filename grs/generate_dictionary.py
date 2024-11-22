import random
from typing import List, Dict
from rs_codec import ReedSolomonCodec, Generalized_Reed_Solomon
from file_handler import FileHandler
from message_generator import generate_unique_binary_strings,binary_string_to_int_list,int_list_to_binary_string,pad_encoded_message


def create_message_codeword_pairs(
    num_messages: int,
    message_length_bits: int = 32,
    codeword_length_bits: int = 48
) -> Dict[str, Dict[str, str]]:
    if message_length_bits % 8 != 0 or codeword_length_bits % 8 != 0:
        raise ValueError("Message and codeword lengths must be multiples of 8 bits")
    
    codec = ReedSolomonCodec(codeword_length_bits // 8, message_length_bits // 8)
    grs_encoder = Generalized_Reed_Solomon(2, codeword_length_bits, message_length_bits, 1, 1, None, False, False)
    
    messages = generate_unique_binary_strings(num_messages, message_length_bits)
    
    data = {}
    for i, message in enumerate(messages, 1):
        encoded_rs = codec.encode(message)
        encoded_rs_bits = codec.symbols_to_bits(encoded_rs)
        
        message_int_list = binary_string_to_int_list(message)
        encoded_grs = grs_encoder.encode(message_int_list)
        encoded_grs = pad_encoded_message(encoded_grs, codeword_length_bits)
        encoded_grs_bits = int_list_to_binary_string(encoded_grs)
        
        data[str(i)] = {
            "message": message,
            "codeword_rs": encoded_rs_bits,
            "codeword_grs": encoded_grs_bits
        }
    
    return data

def verify_codewords(data: Dict[str, Dict[str, str]], message_length_bits: int, codeword_length_bits: int) -> bool:
    codec = ReedSolomonCodec(codeword_length_bits // 8, message_length_bits // 8)
    grs_encoder = Generalized_Reed_Solomon(2, codeword_length_bits, message_length_bits, 1, 1, None, False, False)
    
    for key, value in data.items():
        message = value["message"]
        print(f"Verifying ID {key}...")
        print(f"Original Message: {message}")
        
        # Verify RS codeword
        decoded_rs_symbols = codec.decode(codec.bits_to_symbols(value["codeword_rs"]))
        decoded_rs_bits = codec.symbols_to_bits(decoded_rs_symbols)
        print(f"Decoded RS Message: {decoded_rs_bits}")
        if decoded_rs_bits != message:
            print(f"Verification failed for RS codeword at ID {key}")
            return False
        
        # Verify GRS codeword
        encoded_grs_bits = value["codeword_grs"]
        encoded_grs_int_list = binary_string_to_int_list(encoded_grs_bits)
        decoded_grs = grs_encoder.decode(encoded_grs_int_list)
        decoded_grs_bits = int_list_to_binary_string(decoded_grs)
        print(f"Decoded GRS Message: {decoded_grs_bits[:message_length_bits]}")
        if decoded_grs_bits[:message_length_bits] != message:
            print(f"Verification failed for GRS codeword at ID {key}")
            return False
    
    return True

def main():
    M = 10  # Number of unique binary strings
    K = 32  # Message length
    N = 48  # Codeword length
    output_file = "message_pairs.json"

    # Generate message-codeword pairs
    print(f"Generating {M} messages...")
    data = create_message_codeword_pairs(M, K, N)
    
    # Save to file
    FileHandler.save_to_json(data, output_file)
    print(f"Data saved to {output_file}")
    
    # Verify the generated codewords
    print("Verifying the generated codewords...")
    if verify_codewords(data, K, N):
        print("All codewords verified successfully!")
    else:
        print("Verification failed for some codewords.")
    
    # Print sample
    print("\nSample of generated data:")
    sample_id = "1"
    print(f"ID {sample_id}:")
    print(f"Message:  {data[sample_id]['message']}")
    print(f"Codeword (RS): {data[sample_id]['codeword_rs']}")
    print(f"Codeword (GRS): {data[sample_id]['codeword_grs']}")

if __name__ == "__main__":
    main()