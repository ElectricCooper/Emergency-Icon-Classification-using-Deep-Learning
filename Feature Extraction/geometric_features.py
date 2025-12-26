"""Module for extracting geometric features from images."""

import sys
from pathlib import Path
import numpy as np
# pylint: disable=no-member
import cv2


class GeometricFeatures:
    """Extract geometric features from binary images"""

    @staticmethod
    def calculate_perimeter(image):
        """Calculate the perimeter of contours in a binary image"""
        contours, _ = cv2.findContours(
                        image,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                        )
        return sum(cv2.arcLength(cnt, True) for cnt in contours)

    @staticmethod
    def compute_area(image):
        """Compute area of binary image"""
        return float(cv2.countNonZero(image))

    @staticmethod
    def compute_compactness(perimeter, area):
        """Calculate compactness = (area * 4 * pi) / (perimeter^2)"""
        if perimeter <= 0:
            return 0.0
        return (area * 4 * np.pi) / (perimeter ** 2)

    @staticmethod
    def calculate_convex_area(image):
        """Calculate area using convex hull of the largest contour"""
        contours, _ = cv2.findContours(
                        image,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Combine all contours points to handle multiple part drawings
        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        convex_area = cv2.contourArea(hull)
        area = GeometricFeatures.compute_area(image)

        solidity = area / convex_area if convex_area > 0 else 0

        return {"convex_area": convex_area, "solidity": solidity}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image at {image_path}")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        geo = GeometricFeatures()

        _perimeter = geo.calculate_perimeter(binary)
        _area = geo.compute_area(binary)
        _compactness = geo.compute_compactness(_perimeter, _area)
        _convex_data = geo.calculate_convex_area(binary)

        print(f"Perimeter: {_perimeter}")
        print(f"Area: {_area}")
        print(f"Compactness: {_compactness}")
        print(f"Convex Area: {_convex_data['convex_area']}, " +
              f"Solidity: {_convex_data['solidity']}")
