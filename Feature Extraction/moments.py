"""Module for extracting moment-based features from images"""

import numpy as np
# pylint: disable=no-member
import cv2


class MomentsFeatures:
    """Extract moment-based features"""

    @staticmethod
    def get_hu_moments(image):
        """Calculate Hu moments"""
        # Calculate moments
        moments = cv2.moments(image, True)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Apply log transform to make them more manageable
        hu_moments = -np.sign(hu_moments)*np.log10(np.abs(hu_moments)+1e-10)

        return hu_moments.tolist()

    @staticmethod
    def gravity_center(image):
        """Calculate center of gravity"""
        moments = cv2.moments(image, True)

        if moments['m00'] == 0:
            return (0, 0)

        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']

        return (cx, cy)

    @staticmethod
    def average_centroidal_radius(image):
        """Calculate average centroidal radius"""
        center = MomentsFeatures.gravity_center(image)
        contours, _ = cv2.findContours(
                        image,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                        )

        if not contours:
            return 0.0

        total_distance = 0
        point_count = 0

        for contour in contours:
            for point in contour:
                pt = point[0]
                distance = np.sqrt((pt[0]-center[0])**2 + (pt[1]-center[1])**2)
                total_distance += distance
                point_count += 1

        if point_count == 0:
            return 0.0

        return total_distance / point_count
