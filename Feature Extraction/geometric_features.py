"""Module for extracting geometric features from images."""

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
