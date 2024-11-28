import cv2
import numpy as np

def compute_morb_descriptors(image, keypoints, n_scales=8, scale_factor=1.2):
    """
    Compute MORB descriptors at multiple scales using an image pyramid.

    Args:
        image (numpy.ndarray): Input image.
        keypoints (list): List of keypoints detected in the image.
        n_scales (int): Number of scales for the Gaussian pyramid.
        scale_factor (float): Downsampling factor for the pyramid.

    Returns:
        descriptors (list): List of binary descriptors at each scale for each keypoint.
    """
    pyramid = [image]
    for _ in range(1, n_scales):
        pyramid.append(cv2.pyrDown(pyramid[-1]))

    orb = cv2.ORB_create()
    descriptors = []

    for scale, scaled_image in enumerate(pyramid):
        scaled_kps = [cv2.KeyPoint(kp.pt[0] / (scale_factor ** scale),
                                   kp.pt[1] / (scale_factor ** scale),
                                   kp.size / (scale_factor ** scale))
                      for kp in keypoints]

        _, desc = orb.compute(scaled_image, scaled_kps)
        descriptors.append(desc)

    return descriptors

def cross_scale_matching(descriptors1, descriptors2):
    """
    Match descriptors across scales by finding the minimum Hamming distance.

    Args:
        descriptors1 (list): Descriptors from image 1 (list of arrays for each scale).
        descriptors2 (list): Descriptors from image 2 (list of arrays for each scale).

    Returns:
        matches (list): List of valid matches (queryIdx, trainIdx, min_distance).
    """
    matches = []
    for i, desc1_set in enumerate(descriptors1):
        for j, desc2_set in enumerate(descriptors2):
            if desc1_set is None or desc2_set is None:
                continue

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            raw_matches = bf.knnMatch(desc1_set, desc2_set, k=1)

            for match in raw_matches:
                if len(match) > 0:
                    best_match = match[0]
                    matches.append((best_match.queryIdx, best_match.trainIdx, best_match.distance))

    matches.sort(key=lambda x: x[2])  # Sort by distance
    return matches

# Example Usage
if __name__ == "__main__":
    # Load two images
    image1 = cv2.imread('marker/lasrtag_dictionary/thesis_b_marker_0.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('marker/frame87.png', cv2.IMREAD_GRAYSCALE)
    
    # Detect keypoints using ORB
    orb = cv2.ORB_create()
    keypoints1 = orb.detect(image1, None)
    keypoints2 = orb.detect(image2, None)

    # Compute MORB descriptors
    descriptors1 = compute_morb_descriptors(image1, keypoints1)
    descriptors2 = compute_morb_descriptors(image2, keypoints2)

    # Perform cross-scale matching
    matches = cross_scale_matching(descriptors1, descriptors2)

    # Display matches
    for match in matches[:10]:
        print(f"QueryIdx: {match[0]}, TrainIdx: {match[1]}, Distance: {match[2]}")
        # Draw matches
        matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, 
                                        [cv2.DMatch(_queryIdx=m[0], _trainIdx=m[1], _imgIdx=0, _distance=m[2]) for m in matches[:10]], 
                                        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display the matched image
        cv2.imwrite('morb_match.png', matched_image)