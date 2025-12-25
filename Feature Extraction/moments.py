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
        cx, cy = MomentsFeatures.gravity_center(image)
        contours, _ = cv2.findContours(
                        image,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                        )

        if not contours:
            return 0.0

        all_points = np.vstack(contours).squeeze()

        # Forcing to 2D
        if all_points.ndim == 1:  # if single point it would be 1D
            all_points = np.array([all_points])

        distances = np.sqrt(np.sum((all_points - [cx, cy])**2, axis=1))

        return float(np.mean(distances))
