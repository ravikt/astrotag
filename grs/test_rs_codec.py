import unittest
from .rs_codec import ReedSolomonCodec

class TestReedSolomonCodec(unittest.TestCase):
    def test_symbols_to_bits(self):
        symbols = [147, 37, 171, 125]  
        bits = ReedSolomonCodec.symbols_to_bits(symbols)
        self.assertEqual(bits, '10010011001001011010101101111101', "symbols_to_bits failed")

    def test_bits_to_symbols(self):
        bits = '10010011001001011010101101111101'
        symbols = ReedSolomonCodec.bits_to_symbols(bits)
        self.assertEqual(symbols, [147, 37, 171, 125], "bits_to_symbols failed")

    def test_encode(self):
        codec = ReedSolomonCodec(6, 4)  # Example: n=6, k=4
        message_bits = '10010011001001011010101101111101'
        message_int_list = ReedSolomonCodec.bits_to_symbols(message_bits)
        encoded_message = codec.encode(message_int_list)
        self.assertEqual(len(encoded_message), 6, "Encoded message length is incorrect")
        self.assertIsInstance(encoded_message, list, "Encoded message should be a list")

    def test_decode(self):
        codec = ReedSolomonCodec(6, 4)  # Example: n=6, k=4
        message_bits = '10010011001001011010101101111101'
        message_int_list = ReedSolomonCodec.bits_to_symbols(message_bits)
        encoded_message = codec.encode(message_int_list)
        decoded_message = codec.decode(encoded_message)
        decoded_bits = ReedSolomonCodec.symbols_to_bits(decoded_message)
        self.assertEqual(decoded_bits, message_bits, "Decoded message does not match original message")


if __name__ == "__main__":
    unittest.main()