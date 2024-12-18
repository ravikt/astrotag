# test_descriptor_utils.py
import pytest
import cv2
import numpy as np
from descriptor_utils import detect_and_compute

@pytest.fixture
def sample_image():
    # Create a sample 100x100 grayscale image
    return np.zeros((100, 100), dtype=np.uint8)

def test_descriptor_type_validation(sample_image):
    # Test SIFT descriptor
    keypoints, descriptors = detect_and_compute("SIFT", sample_image)
    assert isinstance(keypoints, list)
    assert descriptors is not None
    
    # Test ORB descriptor
    keypoints, descriptors = detect_and_compute("ORB", sample_image)
    assert isinstance(keypoints, list)
    assert descriptors is not None

    # Test CFD descriptor
    keypoints, descriptors = detect_and_compute("CFD", sample_image)
    assert isinstance(keypoints, list)
    assert descriptors.dtype == np.float32

    # Test LMFD descriptor
    keypoints, descriptors = detect_and_compute("LMFD", sample_image)
    assert isinstance(keypoints, list)
    assert descriptors.dtype == np.float32

def test_invalid_descriptor(sample_image):
    with pytest.raises(ValueError):
        detect_and_compute("INVALID", sample_image)