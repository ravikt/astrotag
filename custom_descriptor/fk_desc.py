import numpy as np
import cv2
import matplotlib.pyplot as plt


def visualize_block_analysis(block):
    plt.figure(figsize=(15, 5))
    
    # Original block with values
    plt.subplot(131)
    plt.imshow(block, cmap='gray')
    plt.title(f'Original Block\nMin: {block.min():.2f}, Max: {block.max():.2f}')
    # Add text annotations for values
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            plt.text(j, i, f'{block[i,j]:.1f}', ha='center', va='center')
    
    # Normalized block
    block_normalized = (block - np.mean(block)) / (np.std(block) + 1e-7)
    plt.subplot(132)
    plt.imshow(block_normalized, cmap='gray')
    plt.title(f'Normalized Block\nMin: {block_normalized.min():.2f}, Max: {block_normalized.max():.2f}')
    # Add text annotations for normalized values
    for i in range(block_normalized.shape[0]):
        for j in range(block_normalized.shape[1]):
            plt.text(j, i, f'{block_normalized[i,j]:.1f}', ha='center', va='center', color='red')
            
    # Add histogram
    plt.subplot(133)
    plt.hist(block_normalized.flatten(), bins=20)
    plt.title('Normalized Values Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Original - Mean: {np.mean(block):.2f}, Std: {np.std(block):.2f}")
    print(f"Normalized - Mean: {np.mean(block_normalized):.2f}, Std: {np.std(block_normalized):.2f}")


def generate_gaussian_pyramid(image, levels):
    """
    Generate a Gaussian Pyramid for the given image.

    :param image: Input image (numpy array)
    :param levels: Number of levels in the pyramid
    :return: List of images in the Gaussian Pyramid
    """
    pyramid = [image]
    for i in range(1, levels):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        # Downsample the image
        image = cv2.pyrDown(blurred)
        pyramid.append(image)
        cv2.imwrite('marker/frame{}.png'.format(i), image)
    return pyramid

def fk_rotation(block):
    # 1. Normalize block to reduce illumination sensitivity
    block_normalized = (block - np.mean(block)) / (np.std(block) + 1e-7)
    
    # 2. Get center pixel
    center_pixel = block_normalized[1, 1]
    
    # 3. Use 8-point neighborhood sampling
    # Define 8 points in neighborhood pattern (clockwise from top-left)
    neighbors = [
        (0,0), (0,1), (0,2),
        (1,2), (2,2),
        (2,1), (2,0),
        (1,0)
    ]
    
    sample_points = []
    for y, x in neighbors:
        sample_points.append(block_normalized[y, x])
    
    # 4. Create binary pattern
    descriptor = (np.array(sample_points) > center_pixel).astype(int)
    
    # 5. Make pattern rotation invariant by finding minimum binary value
    min_pattern = descriptor
    for i in range(len(descriptor)):
        rolled = np.roll(descriptor, i)
        if rolled.tobytes() < min_pattern.tobytes():
            min_pattern = rolled
            
    return min_pattern


def fk_median(block):
    median_val = np.median(block)
    descriptor = (block >= median_val).astype(int)
    return descriptor.flatten()


def fk_combined(image, keypoints, descriptor_size=3, num_levels=3):
    gaussian_pyramid = generate_gaussian_pyramid(image, num_levels)
    scales = [1.0, 0.5, 0.25]
    all_descriptors = []

    

    for level, scale in enumerate(scales):
        scaled_keypoints = [(int(kp[0] * scale), int(kp[1] * scale)) for kp in keypoints]
        descriptors = []
        # debug_image = cv2.cvtColor(gaussian_pyramid[level].copy(), cv2.COLOR_GRAY2BGR)

        for kp in scaled_keypoints:
            x, y = kp
            # cv2.circle(debug_image, (x, y), 2, (0, 255, 0), -1)

            block = gaussian_pyramid[level][max(0, y - 1): y + 2, max(0, x - 1): x + 2]

            if block.shape[0] < descriptor_size or block.shape[1] < descriptor_size:
                continue
            
            # visualize_block_analysis(block)
            rotation_desc = fk_rotation(block)
            median_desc = fk_median(block)
            combined_descriptor = np.hstack([rotation_desc, median_desc])

            descriptors.append(combined_descriptor)
            # descriptors.append(rotation_desc)
        all_descriptors.append(descriptors)
    
        # cv2.imwrite('marker/frame{}.png'.format(level), debug_image)

    # Concatenate descriptors from all levels
    concatenated_descriptors = [np.hstack(desc) for desc in zip(*all_descriptors)]
    
    return np.array(concatenated_descriptors, dtype=np.uint8)

