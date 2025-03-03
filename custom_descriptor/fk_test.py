import cv2
import numpy as np
import matplotlib.pyplot as plt
from read_sig import get_keypoints
from fk_desc import fk_combined

def visualize_block_analysis(block, title_prefix="Block"):
    plt.figure(figsize=(15, 5))
    
    # Original block with values
    plt.subplot(131)
    plt.imshow(block, cmap='gray')
    plt.title(f'{title_prefix}\nMin: {block.min():.2f}, Max: {block.max():.2f}')
    # Add text annotations for values
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            plt.text(j, i, f'{block[i,j]:.1f}', ha='center', va='center')
    
    # Normalized block
    block_normalized = (block - np.mean(block)) / (np.std(block) + 1e-7)
    plt.subplot(132)
    plt.imshow(block_normalized, cmap='gray')
    plt.title(f'Normalized\nMin: {block_normalized.min():.2f}, Max: {block_normalized.max():.2f}')
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


def visualize_keypoints_and_descriptors(image, keypoints, descriptors):
    # Convert image to RGB for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Plot the keypoints on the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    
    # Plot keypoints and their coordinates
    for kp in keypoints:
        x, y = kp[0], kp[1]
        plt.plot(x, y, 'ro')  # Red dot for keypoints
        # Add coordinate text with offset for visibility
        plt.text(x + 5, y + 5, f'({int(x)},{int(y)})', 
                color='yellow', fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.7))
    
    plt.title('Keypoints with Coordinates')
    plt.axis('on')  # Show axes for reference
    plt.grid(True, alpha=0.3)  # Add light grid
    plt.show()
    
    # Display descriptor information
    print("\nKeypoint Descriptors:")
    for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
        print(f"Keypoint {i+1} at ({int(kp[0])},{int(kp[1])}): {desc}")

# Read original image and keypoints
image = cv2.imread('../marker/marker_0.png', cv2.IMREAD_GRAYSCALE)
keypoints = get_keypoints('keypoints_test.txt')

# Create rotated image (90 degrees)
image_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Calculate transformed keypoints for rotated image
height, width = image.shape
keypoints_90 = []

# For 90° clockwise rotation around image center
for kp in keypoints:
    x, y = kp
    
    # Step 1: Translate to origin (center of image)
    x_centered = x - width/2
    y_centered = y - height/2
    
    # Step 2: Rotate 90° clockwise
    # x' = y, y' = -x for 90° clockwise
    x_rotated = y_centered
    y_rotated = -x_centered
    
    # Step 3: Translate back
    new_x = x_rotated + width/2
    new_y = y_rotated + height/2
    
    keypoints_90.append([new_x, new_y])

# Alternative method using OpenCV (this is more reliable)
M = cv2.getRotationMatrix2D((width/2, height/2), -90, 1.0)
keypoints_90 = []
for kp in keypoints:
    pt = np.array([[kp[0], kp[1]]], dtype=np.float32)
    transformed_pt = cv2.transform(np.array([pt]), M)[0][0]
    keypoints_90.append(transformed_pt)

# Compute descriptors for both images
descriptors_0 = fk_combined(image, keypoints)
descriptors_90 = fk_combined(image_90, keypoints_90)

# Measure descriptor similarity using Hamming distance
print("\nComparing descriptors to check rotational invariance:")
print("Keypoint #  |  Hamming Distance  |  Match?")
print("-" * 45)

matches = []
for i in range(len(keypoints)):
    # Get descriptors
    desc1 = descriptors_0[i]
    desc2 = descriptors_90[i]
    
    # Calculate Hamming distance (bits that differ)
    distance = np.count_nonzero(desc1 != desc2)
    
    # Determine if it's a good match (distance < threshold)
    threshold = 0.2 * len(desc1)  # 20% different bits allowed
    is_match = distance < threshold
    
    print(f"{i+1:10d}  |  {distance:17d}  |  {'YES' if is_match else 'NO'}")
    
    if is_match:
        matches.append((i, i, distance))  # Original index, rotated index, distance

# Create a figure for matches only
plt.figure(figsize=(10, 6))

# Create combined image for visualization
h1, w1 = image.shape
h2, w2 = image_90.shape
combined_img = np.zeros((max(h1, h2), w1+w2), dtype=np.uint8)
combined_img[:h1, :w1] = image
combined_img[:h2, w1:w1+w2] = image_90

# Display combined image
plt.imshow(combined_img, cmap='gray')

# Calculate match percentage for title
pct = 100 * len(matches) / len(keypoints)

# Draw match lines with yellow color
for i, j, dist in matches:
    x1, y1 = keypoints[i]
    x2, y2 = keypoints_90[j]
    x2 += w1  # Adjust for second image position
    
    plt.plot([x1, x2], [y1, y2], 'y-', linewidth=1.5)  # Yellow lines
    plt.plot(x1, y1, 'bo', markersize=5)  # Blue for original
    plt.plot(x2, y2, 'ro', markersize=5)  # Red for rotated
    
    # Add keypoint numbers
    plt.text(x1-10, y1-10, f"{i+1}", color='cyan', fontsize=8,
            bbox=dict(facecolor='black', alpha=0.7))
    plt.text(x2+5, y2-10, f"{i+1}", color='cyan', fontsize=8,
            bbox=dict(facecolor='black', alpha=0.7))

# plt.title(f"Descriptor Matches: {len(matches)}/{len(keypoints)} keypoints match ({pct:.1f}% invariant)")
plt.title(f"Descriptor Matches")

plt.axis('off')
plt.tight_layout()
plt.savefig('descriptor_matches.png', dpi=150)
plt.show()

# Print summary
print(f"\nSummary: {len(matches)}/{len(keypoints)} keypoints have rotationally invariant descriptors")
print(f"Rotational invariance: {pct:.1f}%")

# Add after computing descriptors
def analyze_keypoint_descriptors(image, image_90, keypoints, keypoints_90, descriptors_0, descriptors_90):
    """Compare descriptors at corresponding keypoints in original and rotated images"""
    
    # Pick a few keypoints to analyze (e.g., first 3)
    for kp_idx in range(min(3, len(keypoints))):
        print(f"\n===== Analyzing Keypoint {kp_idx+1} =====")
        
        # Get keypoint coordinates
        kp_orig = keypoints[kp_idx]
        kp_rot = keypoints_90[kp_idx]
        
        # Extract image patches around keypoints
        patch_size = 9  # Must be odd
        half_size = patch_size // 2
        
        # Original image patch
        y_orig, x_orig = int(kp_orig[1]), int(kp_orig[0])
        patch_orig = image[
            max(0, y_orig-half_size):min(image.shape[0], y_orig+half_size+1),
            max(0, x_orig-half_size):min(image.shape[1], x_orig+half_size+1)
        ]
        
        # Rotated image patch
        y_rot, x_rot = int(kp_rot[1]), int(kp_rot[0])
        patch_rot = image_90[
            max(0, y_rot-half_size):min(image_90.shape[0], y_rot+half_size+1),
            max(0, x_rot-half_size):min(image_90.shape[1], x_rot+half_size+1)
        ]
        
        # Handle edge cases
        if patch_orig.shape[0] < patch_size or patch_orig.shape[1] < patch_size:
            patch_orig = np.pad(patch_orig, ((0, max(0, patch_size-patch_orig.shape[0])), 
                                           (0, max(0, patch_size-patch_orig.shape[1]))),
                               mode='constant', constant_values=0)
                               
        if patch_rot.shape[0] < patch_size or patch_rot.shape[1] < patch_size:
            patch_rot = np.pad(patch_rot, ((0, max(0, patch_size-patch_rot.shape[0])), 
                                         (0, max(0, patch_size-patch_rot.shape[1]))),
                             mode='constant', constant_values=0)
        
        # Analyze blocks with keypoint coordinates in title
        print(f"\nOriginal Image Block at ({int(kp_orig[0])}, {int(kp_orig[1])}):")
        visualize_block_analysis(patch_orig, f"Original - Keypoint {kp_idx+1} at ({int(kp_orig[0])}, {int(kp_orig[1])})")
        
        print(f"\nRotated Image Block at ({int(kp_rot[0])}, {int(kp_rot[1])}):")
        visualize_block_analysis(patch_rot, f"Rotated - Keypoint {kp_idx+1} at ({int(kp_rot[0])}, {int(kp_rot[1])})")
        
        # Compare descriptors
        desc_orig = descriptors_0[kp_idx]
        desc_rot = descriptors_90[kp_idx]
        
        print("\nDescriptor Comparison:")
        print(f"Original descriptor: {desc_orig}")
        print(f"Rotated descriptor: {desc_rot}")
        
        # Calculate differences
        diff = desc_orig != desc_rot
        diff_count = np.count_nonzero(diff)
        diff_pct = 100 * diff_count / len(desc_orig)
        
        print(f"Different bits: {diff_count}/{len(desc_orig)} ({diff_pct:.1f}%)")
        print(f"Matching?: {'YES' if diff_count < 0.2 * len(desc_orig) else 'NO'}")

# Call the analysis function
analyze_keypoint_descriptors(
    image, image_90, 
    keypoints, keypoints_90, 
    descriptors_0, descriptors_90
)

