"""POI (Points of Interest) detection module"""

# pylint: disable=no-member
import cv2


class POIDetection:
    """Detect points of interest, here corners only"""

    @staticmethod
    def detect_all_corners(image, epsilon_factor=0.02):
        """Detect corners with significant angle changes
            by simplifying contours into polygons
        """
        contours, _ = cv2.findContours(
                        image,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

        total_corners = 0
        for cnt in contours:
            # Espilon : max distance from contour to approximated shape
            perimeter = cv2.arcLength(cnt, True)
            epsilon = epsilon_factor * perimeter

            # Approximate shape with a polygon
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Number of vertices in polygon is the number of corners
            total_corners += len(approx)

        return total_corners

    @staticmethod
    def detect_sharp_corners(image,
                             max_corners=20,
                             quality_level=0.01,
                             min_distance=10):
        """Find most sharp corners using Shi-Tomasi method"""
        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance
        )

        return len(corners) if corners is not None else 0
