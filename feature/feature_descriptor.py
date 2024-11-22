import cv2
import numpy as np
from typing import List, Tuple
import numpy as np
from typing import List, Tuple, Optional
import cv2

class TriMeshDetector:
    """
    Feature detector for binary triangular mesh patterns.
    Handles multiple resolutions and detects vertices in triangular grids.
    """
    
    def __init__(self, min_resolution: int = 32, max_resolution: int = 512):
        """
        Initialize detector with resolution range.
        
        Args:
            min_resolution: Minimum image width/height to process
            max_resolution: Maximum image width/height to process
        """
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        
        # Create triangular junction kernels
        self.kernels = self._create_tri_kernels()

    def _create_tri_kernels(self) -> List[np.ndarray]:
        """Create kernels for detecting triangular vertices."""
        # Basic Y-junction kernel (120-degree angles)
        k1 = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8)
        
        # Create rotated versions for different orientations
        kernels = []
        for angle in [0, 60, 120]:
            center = (k1.shape[1] // 2, k1.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(k1, rotation_matrix, (k1.shape[1], k1.shape[0]))
            kernels.append((rotated > 0).astype(np.uint8))
            
        return kernels

    def _process_single_resolution(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Process image at a single resolution.
        
        Args:
            image: Binary input image
            
        Returns:
            List of (y, x) vertex coordinates
        """
        vertices = []
        
        # Ensure image is binary
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Apply each orientation kernel
        accumulated_response = np.zeros_like(binary, dtype=np.float32)
        
        for kernel in self.kernels:
            # Match kernel pattern
            response = cv2.matchTemplate(binary, kernel, cv2.TM_CCORR_NORMED)
            response = cv2.resize(response, (binary.shape[1], binary.shape[0]))
            accumulated_response += response
        
        # Find local maxima in accumulated response
        maxima = cv2.dilate(accumulated_response, None)
        peaks = (accumulated_response == maxima) & (accumulated_response > 0.7)
        y_coords, x_coords = np.nonzero(peaks)
        
        # Filter closely spaced vertices
        min_distance = max(3, image.shape[0] // 50)  # Adaptive minimum distance
        filtered_vertices = []
        
        for y, x in zip(y_coords, x_coords):
            # Check if this point is far enough from all accepted points
            if not filtered_vertices or all(
                abs(y - fy) > min_distance or abs(x - fx) > min_distance 
                for fy, fx in filtered_vertices
            ):
                filtered_vertices.append((y, x))
        
        return filtered_vertices

    def detect(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect vertices in triangular mesh at multiple resolutions.
        
        Args:
            image: Grayscale input image
            
        Returns:
            List of (y, x) vertex coordinates
        """
        all_vertices = []
        current_image = image.copy()
        
        # Process image at multiple resolutions
        while min(current_image.shape[:2]) >= self.min_resolution:
            # Detect vertices at current resolution
            vertices = self._process_single_resolution(current_image)
            
            # Scale coordinates back to original image size
            scale_y = image.shape[0] / current_image.shape[0]
            scale_x = image.shape[1] / current_image.shape[1]
            
            scaled_vertices = [
                (int(y * scale_y), int(x * scale_x))
                for y, x in vertices
            ]
            
            all_vertices.extend(scaled_vertices)
            
            # Resize image for next iteration
            current_image = cv2.resize(
                current_image, 
                (current_image.shape[1] // 2, current_image.shape[0] // 2)
            )
        
        # Remove duplicates and near-duplicates
        final_vertices = self._remove_duplicates(all_vertices, threshold=5)
        
        return final_vertices

    def _remove_duplicates(self, vertices: List[Tuple[int, int]], 
                         threshold: int) -> List[Tuple[int, int]]:
        """Remove duplicate and near-duplicate vertices."""
        if not vertices:
            return []
            
        filtered = []
        for vertex in vertices:
            # Check if this vertex is far enough from all accepted vertices
            if not filtered or all(
                abs(vertex[0] - v[0]) > threshold or 
                abs(vertex[1] - v[1]) > threshold 
                for v in filtered
            ):
                filtered.append(vertex)
        
        return filtered

    def analyze_mesh(self, image: np.ndarray, 
                    vertices: List[Tuple[int, int]]) -> dict:
        """
        Analyze mesh properties.
        
        Args:
            image: Input image
            vertices: Detected vertex coordinates
            
        Returns:
            Dictionary with mesh properties
        """
        if len(vertices) < 3:
            return {}
            
        # Calculate typical edge lengths
        edges = []
        for i, (y1, x1) in enumerate(vertices[:-1]):
            for y2, x2 in vertices[i+1:]:
                dist = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                edges.append(dist)
        
        if not edges:
            return {}
            
        # Analyze edge length distribution
        edges = np.array(edges)
        return {
            'avg_edge_length': float(np.mean(edges)),
            'min_edge_length': float(np.min(edges)),
            'max_edge_length': float(np.max(edges)),
            'vertex_count': len(vertices),
            'estimated_resolution': float(np.median(edges))
        }

def example_usage():
    # Create a sample triangular mesh image
    size = 200
    # image = np.zeros((size, size), dtype=np.uint8)
    image = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    # Draw some triangular grid lines
    for i in range(0, size, 20):
        angle = 60 * np.pi / 180
        x1 = i
        y1 = 0
        x2 = int(i + size * np.cos(angle))
        y2 = int(size * np.sin(angle))
        cv2.line(image, (x1, y1), (x2, y2), 255, 1)
        cv2.line(image, (x1, y1), (x1, size-1), 255, 1)
    
    # Initialize detector
    detector = TriMeshDetector()
    
    # Detect vertices
    vertices = detector.detect(image)
    
    # Analyze mesh
    properties = detector.analyze_mesh(image, vertices)
    print("Mesh properties:", properties)
    
    # Visualize results
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for y, x in vertices:
        cv2.circle(vis_image, (x, y), 2, (0, 0, 255), -1)
    
    cv2.imwrite("output.png", vis_image)




if __name__ == '__main__':
    example_usage()
