import cv2
import numpy as np
import mfd_descriptor  
from lmfd_extractor import LMFD_DescriptorExtractor

def detect_and_compute(descriptor_type, image):
    if descriptor_type == "SIFT":
        detector = cv2.SIFT_create()
    elif descriptor_type == "ORB":
        detector = cv2.ORB_create()
    elif descriptor_type == "MFD":
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(image, None)
        descriptors = mfd_descriptor.mfd_combined_descriptor(image, keypoints)
        return keypoints, descriptors.astype(np.float32)
    elif descriptor_type == "LMFD":
        lmfd = LMFD_DescriptorExtractor()
        keypoints, descriptors = lmfd.lmfd_detector(image)
        return keypoints, descriptors
    else:
        raise ValueError("Unsupported descriptor type")

    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

