import cv2
import numpy as np

def detect_quadrilaterals(image_path):
    """
    Detect quadrilaterals in an image using a combination of contour detection
    and Hough transform, as described in the paper.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        list: List of detected quadrilateral contours
        image: Original image with detected quadrilaterals drawn
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    # Create a copy for drawing results
    output = image.copy()
    
    # Preprocessing steps as mentioned in the paper
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.png', gray)
    
    # 2. Normalize and enhance edges using histogram equalization
    # Skipping this step for now
    
    # Skipping edge detection step for now
    gradient_magnitude = gray  # Use the grayscale image directly as a placeholder
    
    # 4. Adaptive thresholding with inversion to get binary image
    binary = cv2.adaptiveThreshold(gradient_magnitude, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite('binary.png', binary)
    
    # 5. Find contours (as mentioned in section 3.2.1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_quadrilaterals = []
    
    for idx, contour in enumerate(contours):
        # Skip very small contours
        if cv2.contourArea(contour) < 100:
            continue
            
        # Convert contour points to a format suitable for Hough transform
        contour_img = np.zeros_like(binary)
        cv2.drawContours(contour_img, [contour], -1, 255, 1)
        
        # Apply Hough transform on the contour
        lines = cv2.HoughLines(contour_img, 1, np.pi/180, threshold=50)
        
        if lines is not None and len(lines) >= 4:
            # Convert lines from polar to Cartesian coordinates
            cartesian_lines = []
            for rho, theta in lines[:8, 0]:  # Consider up to 8 strongest lines
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cartesian_lines.append(((x1, y1), (x2, y2)))
            
            # Draw the detected lines on the contour image
            for line in cartesian_lines:
                cv2.line(contour_img, line[0], line[1], 255, 1)
            
            # Save the contour image with lines
            cv2.imwrite(f'contour_lines_{idx}.png', contour_img)
            
            # Find parallel line pairs
            parallel_pairs = []
            for i in range(len(cartesian_lines)):
                for j in range(i + 1, len(cartesian_lines)):
                    # Calculate angle between lines
                    angle1 = np.arctan2(cartesian_lines[i][1][1] - cartesian_lines[i][0][1],
                                      cartesian_lines[i][1][0] - cartesian_lines[i][0][0])
                    angle2 = np.arctan2(cartesian_lines[j][1][1] - cartesian_lines[j][0][1],
                                      cartesian_lines[j][1][0] - cartesian_lines[j][0][0])
                    angle_diff = abs(angle1 - angle2)
                    
                    # If lines are parallel (angle difference close to 0 or pi)
                    if angle_diff < 0.1 or abs(angle_diff - np.pi) < 0.1:
                        parallel_pairs.append((i, j))
            
            # If we found at least 2 pairs of parallel lines
            if len(parallel_pairs) >= 2:
                # Approximate the contour to get vertices
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If the approximated contour has 4 vertices
                if len(approx) == 4:
                    # Calculate aspect ratio (as mentioned in section 3.2.2.1)
                    rect = cv2.minAreaRect(approx)
                    width = rect[1][0]
                    height = rect[1][1]
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    # Add to detected quadrilaterals if aspect ratio is reasonable
                    if 0.5 < aspect_ratio < 5.0:
                        detected_quadrilaterals.append(approx)
                        cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)
    
    return detected_quadrilaterals, output

def draw_results(image_path, save_path=None):
    """
    Process an image and draw detected quadrilaterals.
    
    Args:
        image_path (str): Path to input image
        save_path (str, optional): Path to save the output image
    """
    try:
        quads, result_image = detect_quadrilaterals(image_path)
        print(f"Found {len(quads)} quadrilaterals")
        
        if save_path:
            cv2.imwrite(save_path, result_image)
            print(f"Results saved to {save_path}")
            
        return result_image
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None
    
# Example usage 
if __name__ == "__main__":
    draw_results('test_marker.png', 'output.png')