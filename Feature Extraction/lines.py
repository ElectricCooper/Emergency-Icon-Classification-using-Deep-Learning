"""Line detection module"""

import numpy as np
# pylint: disable=no-member
import cv2


class LineDetection:
    """Detect and regroup lines in images"""

    @staticmethod
    def detect_lines(image):
        """Detect lines using Hough Transform"""
        # Apply Canny edge detection
        lines = cv2.HoughLinesP(
            image,
            rho=1,
            theta=np.pi/180,
            threshold=10,      # min intersections to detect line
            minLineLength=10,  # min length of line
            maxLineGap=5       # max gap between pixels
        )

        if lines is None:
            return []

        # return list of x1,y1,x2,y2
        return [line[0] for line in lines]

    @staticmethod
    def classify_directions(lines):
        """Classify lines into horizontal, vertical, and diagonal"""
        results = {'horizontal': 0, 'vertical': 0, 'diag1': 0, 'diag2': 0}

        for x1, y1, x2, y2 in lines:
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx)) % 180  # Normalize to 0-180

            if 67.5 <= angle <= 112.5:
                results['horizontal'] += 1
            elif angle <= 22.5 or angle >= 157.5:
                results['vertical'] += 1
            elif 22.5 < angle < 67.5:
                results['diag1'] += 1
            else:
                results['diag2'] += 1

        return results
