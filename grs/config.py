# File: config.py
"""Configuration settings for the Reed-Solomon encoder program."""

class Config:
    # Default parameters
    DEFAULT_MESSAGE_LENGTH = 32  # bits
    DEFAULT_CODEWORD_LENGTH = 48  # bits
    DEFAULT_NUM_MESSAGES = 10
    
    # File settings
    OUTPUT_DIRECTORY = "output"
    DEFAULT_OUTPUT_FILENAME = "message_codeword_pairs.json"
    
    # Validation
    MIN_MESSAGE_LENGTH = 8  # minimum message length in bits
    MAX_MESSAGE_LENGTH = 1024  # maximum message length in bits
    
    @classmethod
    def validate_parameters(cls, message_length: int, codeword_length: int, num_messages: int):
        """Validate input parameters."""
        if message_length < cls.MIN_MESSAGE_LENGTH:
            raise ValueError(f"Message length must be at least {cls.MIN_MESSAGE_LENGTH} bits")
            
        if message_length > cls.MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message length cannot exceed {cls.MAX_MESSAGE_LENGTH} bits")
            
        if codeword_length <= message_length:
            raise ValueError("Codeword length must be greater than message length")
            
        if message_length % 8 != 0 or codeword_length % 8 != 0:
            raise ValueError("Message and codeword lengths must be multiples of 8 bits")
            
        if num_messages < 1:
            raise ValueError("Number of messages must be positive")