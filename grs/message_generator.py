# File: message_generator.py
import random
from typing import List, Dict
from rs_codec import ReedSolomonCodec

def generate_unique_binary_strings(num_strings=10, message_length=32) -> List[str]:
    unique_strings = set()
    while len(unique_strings) < num_strings:
        binary_string = ''.join(random.choice('01') for _ in range(message_length))
        unique_strings.add(binary_string)
    return list(unique_strings)

def binary_string_to_int_list(binary_string: str) -> List[int]:
    return [int(bit) for bit in binary_string]

def int_list_to_binary_string(int_list: List[int]) -> str:
    return ''.join(str(bit) for bit in int_list)

def pad_encoded_message(encoded_message: List[int], expected_length: int) -> List[int]:
    if len(encoded_message) < expected_length:
        padding = [0] * (expected_length - len(encoded_message))
        encoded_message.extend(padding)
    return encoded_message

    
def create_message_codeword_pairs(
    num_messages: int,
    message_length_bits: int = 32,
    codeword_length_bits: int = 48
) -> Dict[str, Dict[str, str]]:
    """
    Generate message-codeword pairs and return them as a dictionary.
    
    Args:
        num_messages: Number of unique messages to generate
        message_length_bits: Length of each message in bits
        codeword_length_bits: Length of each codeword in bits
    
    Returns:
        Dictionary with IDs mapping to message-codeword pairs
    """
    # Validate input parameters
    if message_length_bits % 8 != 0 or codeword_length_bits % 8 != 0:
        raise ValueError("Message and codeword lengths must be multiples of 8 bits")
    
    # Initialize the Reed-Solomon codec
    codec = ReedSolomonCodec(codeword_length_bits // 8, message_length_bits // 8)
    
    # Generate unique messages
    messages = generate_unique_messages(num_messages, message_length_bits)
    
    # Create the dictionary
    data = {}
    for i, message in enumerate(messages, 1):
        # Encode the message
        encoded = codec.encode(message)
        encoded_bits = codec.symbols_to_bits(encoded)
        
        # Add to dictionary with ID as key
        data[str(i)] = {
            "message": message,
            "codeword": encoded_bits
        }
    
    return data