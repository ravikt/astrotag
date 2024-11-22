import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_scale_pyramid(image, levels):
    pyramid = [image]
    for i in range(1, levels):
        scaled_image = cv2.pyrDown(pyramid[-1])
        pyramid.append(scaled_image)
    return pyramid

def display_pyramid(pyramid):
    for i, img in enumerate(pyramid):
        plt.subplot(1, len(pyramid), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Level {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Load an image with a transparent background
image_path = 'qr.png'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Create a scale pyramid with a specified number of levels
levels = 8  # You can change this to create more or fewer levels
pyramid = create_scale_pyramid(image, levels)

# Display the scale pyramid
display_pyramid(pyramid)