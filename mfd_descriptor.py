import numpy as np
import cv2

# def MFD_symbolic_descriptor(block):
#     center_pixel = block[1, 1]
#     descriptor = (block > center_pixel).astype(int)
#     return descriptor.flatten()


def mfd_symbolic_descriptor(block):
    # 1. Normalize block to reduce illumination sensitivity
    block_normalized = (block - np.mean(block)) / (np.std(block) + 1e-7)
    
    # 2. Get center pixel
    center_pixel = block_normalized[1, 1]
    
    # 3. Use circular sampling pattern for rotation invariance
    # Define 8 points in circular pattern
    radius = 1
    angles = np.arange(0, 2*np.pi, 2*np.pi/8)
    sample_points = []
    
    center_y, center_x = 1, 1
    for angle in angles:
        y = center_y + radius * np.sin(angle)
        x = center_x + radius * np.cos(angle)
        # Bilinear interpolation for non-integer coordinates
        y0, y1 = int(np.floor(y)), int(np.ceil(y))
        x0, x1 = int(np.floor(x)), int(np.ceil(x))
        wy1 = y - y0
        wy0 = 1 - wy1
        wx1 = x - x0
        wx0 = 1 - wx1
        
        val = block_normalized[y0:y1+1, x0:x1+1].flatten()
        if len(val) == 4:
            interpolated = val[0]*wy0*wx0 + val[1]*wy0*wx1 + \
                         val[2]*wy1*wx0 + val[3]*wy1*wx1
            sample_points.append(interpolated)
    
    # 4. Create rotation invariant binary pattern
    descriptor = (np.array(sample_points) > center_pixel).astype(int)
    
    # 5. Make pattern rotation invariant by finding minimum binary value
    min_pattern = descriptor
    for i in range(len(descriptor)):
        rolled = np.roll(descriptor, i)
        if rolled.tobytes() < min_pattern.tobytes():
            min_pattern = rolled
            
    return min_pattern


# def mfd_mean_descriptor(block):
#     mean_val = np.mean(block)
#     descriptor = (block >= mean_val).astype(int)
#     return descriptor.flatten()
def mfd_median_descriptor(block):
    median_val = np.median(block)
    descriptor = (block >= median_val).astype(int)
    return descriptor.flatten()

# def mfd_centroid_descriptor(block):
#     h, w = block.shape
#     y, x = np.meshgrid(np.arange(w), np.arange(h))
#     centroid_y = int(np.sum(y * block) / np.sum(block))
#     centroid_x = int(np.sum(x * block) / np.sum(block))
    
#     descriptor = np.zeros_like(block)
#     descriptor[centroid_y, centroid_x] = 1
#     return descriptor.flatten()

def mfd_combined_descriptor(image, keypoints, descriptor_size=3):
    descriptors = []
    for kp in keypoints:
        # x, y = int(kp.pt[0]), int(kp.pt[1]) # For FAST Keypoint
        x, y = int(kp[0]), int(kp[1])
        block = image[max(0, y - 1): y + 2, max(0, x - 1): x + 2]
        
        if block.shape[0] < descriptor_size or block.shape[1] < descriptor_size:
            continue
        
        symbolic_desc = mfd_symbolic_descriptor(block)
        # mean_desc = mfd_mean_descriptor(block)
        mean_desc = mfd_median_descriptor(block)
        # centroid_desc = mfd_centroid_descriptor(block)
        
        # combined_descriptor = np.hstack([symbolic_desc, mean_desc, centroid_desc])
        combined_descriptor = np.hstack([symbolic_desc, mean_desc])

        descriptors.append(combined_descriptor)
    
    print(len(descriptors))
    # print(descriptors)
    # Convert list of descriptors to numpy array of type uint8
    return np.array(descriptors, dtype=np.uint8)
