import cv2
import numpy as np

def extract_features(img_path, keypoints, descriptor):
    """Compute descriptors for given keypoints in an image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute descriptors using the provided descriptor
    keypoints, descriptors = descriptor.compute(img, keypoints)
    
    return keypoints, descriptors

def compute_matches(descriptors1, descriptors2, matcher, descriptor_type='float', ratio_thresh=0.75):
    """Compute matches between two sets of descriptors."""
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptors cannot be None")
    
    print(f"Descriptor1 shape: {descriptors1.shape}, type: {descriptors1.dtype}")
    print(f"Descriptor2 shape: {descriptors2.shape}, type: {descriptors2.dtype}")
    
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    pts1 = []
    pts2 = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if descriptor_type == 'float':
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                pts1.append([m.queryIdx])
                pts2.append([m.trainIdx])
        elif descriptor_type == 'binary':
            good_matches.append(m)
            pts1.append([m.queryIdx])
            pts2.append([m.trainIdx])
    return good_matches, pts1, pts2

def compute_repeatability(ref_img_path, test_img_path, keypoints, ref_descriptor, test_descriptor, matcher, descriptor_type='float'):
    """Compute repeatability of keypoints between reference and test images."""
    # Compute descriptors for reference image using the provided keypoints
    kp1, desc1 = extract_features(ref_img_path, keypoints, ref_descriptor)
    
    # Extract keypoints and compute descriptors for test image
    kp2, desc2 = extract_features(test_img_path, keypoints, test_descriptor)
    
    # Debug prints for keypoints and descriptors
    print(f"Reference image: {len(kp1)} keypoints, {desc1.shape} descriptors")
    print(f"Test image: {len(kp2)} keypoints, {desc2.shape} descriptors")
    
    # Compute matches
    good_matches, pts1, pts2 = compute_matches(desc1, desc2, matcher, descriptor_type)
    
    # Extract matched keypoints
    matched_kp1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    matched_kp2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography
    H, mask = cv2.findHomography(matched_kp1, matched_kp2, cv2.RANSAC)
    
    # Transform points from reference image
    all_pts1 = np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2)
    pts1_transformed = cv2.perspectiveTransform(all_pts1, H)
    
    # Compute repeatability
    repeatability = np.sum(mask) / len(kp1)
    
    return repeatability

# Example usage
if __name__ == "__main__":
    ref = 'ref_marker.jpg'
    test = 'original.png'
    
    # Initialize descriptors and matcher
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()
    bf_matcher_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf_matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Extract keypoints from reference image using FAST
    img = cv2.imread(ref, cv2.IMREAD_GRAYSCALE)
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(img, None)
    
    # Compute repeatability for SIFT
    repeatability_sift = compute_repeatability(ref, test, keypoints, sift, sift, bf_matcher_sift, descriptor_type='float')
    print(f"SIFT Repeatability: {repeatability_sift}")
    
    # Compute repeatability for ORB
    repeatability_orb = compute_repeatability(ref, test, keypoints, orb, orb, bf_matcher_orb, descriptor_type='binary')
    print(f"ORB Repeatability: {repeatability_orb}")