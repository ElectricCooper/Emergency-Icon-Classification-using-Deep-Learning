""" Module for analyzing diagonal properties of drawings """

import numpy as np
# pylint: disable=no-member
import cv2


class DiagonalAnalysis:
    """Analyze diagonal properties of drawings"""

    @staticmethod
    def analyze_diagonal(image):
        """Find and measure diagonal of the drawing"""
        # Safe checking image is 1-channel (binary from process.py)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find coordinates of all non-zero (white) pixels
        coords = cv2.findNonZero(image)

        if coords is None:
            return {'length': 0, 'angle': 0}

        _, _, w, h = cv2.boundingRect(coords)  # Bounding box

        # Calculate diagonal length and angle
        diagonal_length = np.sqrt(w**2 + h**2)
        diagonal_angle = np.degrees(np.arctan2(h, w))

        return {
            'length': round(diagonal_length, 3),
            'angle': round(diagonal_angle, 3)
        }
