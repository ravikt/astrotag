import cv2
import numpy as np
import descriptor_utils as du
from plot_utils import plot_distance_distribution

def evaluate_descriptor(descriptor, image1, image2):
    try:
        kp1, desc1 = du.detect_and_compute(descriptor, image1)
        kp2, desc2 = du.detect_and_compute(descriptor, image2)

        # Convert descriptors to correct type
        if descriptor == "MFD":
            desc1 = desc1.astype(np.uint8)
            desc2 = desc2.astype(np.uint8)
        
        # Matcher selection
        if descriptor in ["SIFT", "LMFD"]:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        correct_matches = sum(1 for m in matches if m.distance < 50)
        accuracy = correct_matches / len(matches) if matches else 0
        match_time = sum(m.distance for m in matches) / len(matches) if matches else 0

        return accuracy, match_time
    except Exception as e:
        print(f"Error processing {descriptor}: {str(e)}")
        return 0, 0

# def compute_repeatability(descriptor, image1, transformed_image):
#     try:
#         kp1, desc1 = du.detect_and_compute(descriptor, image1)
#         kp2, desc2 = du.detect_and_compute(descriptor, transformed_image)

#         # Convert descriptors to correct type
#         if descriptor == "MFD":
#             desc1 = desc1.astype(np.uint8)
#             desc2 = desc2.astype(np.uint8)

#         # Matcher selection
#         if descriptor in ["SIFT", "LMFD"]:
#             matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#         else:
#             matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#         matches = matcher.match(desc1, desc2)
#         inlier_count = sum(1 for m in matches if m.distance < 50)
#         repeatability = inlier_count / len(matches) if matches else 0
#         return repeatability
#     except Exception as e:
#         print(f"Error computing repeatability for {descriptor}: {str(e)}")
#         return 0

# metrics.py
def compute_repeatability(descriptor, image1, transformed_image):
    try:
        kp1, desc1 = du.detect_and_compute(descriptor, image1)
        kp2, desc2 = du.detect_and_compute(descriptor, transformed_image)
    
        # Convert descriptors to appropriate type if necessary
        if descriptor == "MFD":
            desc1 = desc1.astype(np.float32)
            desc2 = desc2.astype(np.float32)
    
        # Choose matcher based on descriptor type
        if descriptor in ["SIFT", "LMFD", "MFD"]:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
        matches = matcher.match(desc1, desc2)
    
        # Analyze match distances
        distances = [m.distance for m in matches]
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
    
        # Adjust threshold dynamically
        threshold = avg_distance + std_distance  # Or any other heuristic
    
        inlier_count = sum(1 for m in matches if m.distance < threshold)
        repeatability = inlier_count / len(matches) if matches else 0
        
        # plot_distance_distribution(distances, descriptor)    
        
        return repeatability
    except Exception as e:
        print(f"Error computing repeatability for {descriptor}: {str(e)}")
        return 0

def visualize_matches(descriptor, image1, image2, output_path=None):
    """
    Draw and save matches between two images for a given descriptor.
    
    Args:
        descriptor (str): Descriptor type ("SIFT", "ORB", "MFD", "LMFD")
        image1 (ndarray): First image
        image2 (ndarray): Second image
        output_path (str, optional): Path to save visualization. If None, shows interactive window
    """
    try:
        # Get keypoints and descriptors
        kp1, desc1 = du.detect_and_compute(descriptor, image1)
        kp2, desc2 = du.detect_and_compute(descriptor, image2)

        # Convert descriptors to correct type
        if descriptor == "MFD":
            desc1 = desc1.astype(np.uint8)
            desc2 = desc2.astype(np.uint8)

        # Matcher selection
        if descriptor in ["SIFT", "LMFD"]:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Filter good matches
        good_matches = [m for m in matches if m.distance < 50]
        
        # Draw matches
        match_img = cv2.drawMatches(
            image1, kp1, 
            image2, kp2, 
            good_matches[:30],  # Show top 30 matches
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        if output_path:
            cv2.imwrite(output_path, match_img)
        else:
            cv2.imshow(f'{descriptor} Matches', match_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return len(good_matches)
    except Exception as e:
        print(f"Error visualizing matches for {descriptor}: {str(e)}")
        return 0
