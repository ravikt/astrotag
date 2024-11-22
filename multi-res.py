import os
import cv2
import re

# Define the input and output directories
input_dir = '/home/ravikt/repositories/lasrtag/results/original/input_images'
output_dir = '/home/ravikt/repositories/lasrtag/results/double/input_images'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize a counter
count = 0
max_images = 1000

# Define the pattern for the filenames
pattern = re.compile(r'^frame\d{1,3}\.png$')

# Iterate over all files in the input directory
for filename in sorted(os.listdir(input_dir)):
    if count >= max_images:
        break
    if pattern.match(filename):
        # Read the image file
        img = cv2.imread(os.path.join(input_dir, filename))
        if img is not None:
            # Calculate the new size
            new_size = (img.shape[1]*2 , img.shape[0]*2 )
            # Resize the image
            resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            # Save the resized image to the output directory
            cv2.imwrite(os.path.join(output_dir, filename), resized_img)
            # Increment the counter
            count += 1

print("First 1000 images have been resized and saved to the output folder.")