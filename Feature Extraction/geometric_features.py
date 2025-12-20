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
        perimeter = 0.0
        for contour in contours:
            perimeter += cv2.arcLength(contour, True)
        return perimeter

    @staticmethod
    def compute_area(binary_img):
        """Compute area using morphological operations"""
        kernel_size = 100
        kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (kernel_size, kernel_size)
                    )

        # Adding padding
        padding = kernel_size // 2
        padded_img = cv2.copyMakeBorder(
                        binary_img,
                        padding,
                        padding,
                        padding,
                        padding,
                        cv2.BORDER_CONSTANT,
                        value=255)

        # Morphological operations
        eroded_img = cv2.erode(padded_img, kernel)
        dilated_img = cv2.dilate(eroded_img, kernel)

        # Removing padding
        binary_img = dilated_img[padding:padding + binary_img.shape[0],
                                 padding:padding + binary_img.shape[1]]

        # Invert and count
        binary_img = cv2.bitwise_not(binary_img)
        area = cv2.countNonZero(binary_img)

        return area

    @staticmethod
    def compute_compactness(perimeter, area):
        """Calculate compactness = (area * 4 * pi) / (perimeter^2)"""
        if perimeter == 0:
            return 0.0
        return (area * 4 * np.pi) / (perimeter * perimeter)

    @staticmethod
    def calculate_convex_area(image):
        """Calculate area using convex hull of the largest contour"""

        # Find contours
        contours, _ = cv2.findContours(
                        image,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # get all points
        all_points = []
        for contour in contours:
            all_points.extend(contour.reshape(-1, 2))

        if len(all_points) < 3:
            return 0.0

        all_points = np.array(all_points)

        # Compute convex hull
        hull = cv2.convexHull(all_points)
        area = cv2.contourArea(hull)

        return area
