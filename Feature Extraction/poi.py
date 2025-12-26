"""POI (Points of Interest) detection module"""

import sys
from pathlib import Path
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

        corner_count = 0
        approxs = []
        for cnt in contours:
            # Espilon : max distance from contour to approximated shape
            perimeter = cv2.arcLength(cnt, True)
            epsilon = epsilon_factor * perimeter

            # Approximate shape with a polygon
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approxs.append(approx)

            # Number of vertices in polygon is the number of corners
            corner_count += len(approx)

        return corner_count, approxs

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
        count = len(corners) if corners is not None else 0
        return count, corners


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

        detector = POIDetection()

        total_corners, approximations = detector.detect_all_corners(binary)

        total_sharp_corners, sh_corners = detector.detect_sharp_corners(binary)

        print(f"Corners count via polygon approximation: {total_corners}")
        print(f"Sharp corners count: {total_sharp_corners}")

        output_img = img.copy()

        for polygon in approximations:
            cv2.polylines(output_img,
                          [polygon],
                          isClosed=True,
                          color=(0, 255, 0),
                          thickness=2)

        if sh_corners is not None:
            for corner in sh_corners:
                x, y = corner.ravel()
                cv2.circle(output_img, (int(x), int(y)), 5, (0, 0, 255), -1)

        cv2.imshow("POI detection (green: structural, red: sharp) ",
                   output_img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
