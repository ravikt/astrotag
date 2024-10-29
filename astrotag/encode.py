from cryptography.fernet import Fernet

# Generate a key (do this once and save the key securely)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Read the file contents
with open('triangle_list.txt', 'rb') as file:
    file_data = file.read()

# Encrypt the data
encrypted_data = cipher_suite.encrypt(file_data)

# Print the key and encrypted data (save the key securely)
print(f"Key: {key.decode()}")
print(f"Encrypted Data: {encrypted_data.decode()}")
