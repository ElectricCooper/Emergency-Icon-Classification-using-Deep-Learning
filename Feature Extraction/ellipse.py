"""Ellipse detection module"""

import sys
from pathlib import Path
import numpy as np
# pylint: disable=no-member
import cv2


class EllipseDetection:
    """Detect ellipses in images"""

    @staticmethod
    def detect_ellipses(image, min_ratio=0.3, min_area=20):
        """Detect ellipses in an image"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Close small gaps in circles
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(image, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

        detected_ellipses = []
        for contour in contours:
            area = cv2.contourArea(contour)
            print(f"Contour area: {area}")
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity > 0.92 and area >= min_area:

                ellipse = cv2.fitEllipse(contour)
                # Aspect ratio
                w, h = ellipse[1]
                ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0

                # Circularity (making sure it's not a random pt scatter)
                perimeter = cv2.arcLength(contour, True)
                circularity = 0
                if perimeter > 0:
                    circularity = (4*np.pi*area)/(perimeter**2)

                if ratio > min_ratio and circularity > 0.75:  # Thresholds
                    detected_ellipses.append(ellipse)

        # Sort (descending) by area
        detected_ellipses.sort(key=lambda e: e[1][0] * e[1][1], reverse=True)
        # Filter overlapping ellipses and select top 2
        # (we don't expect more than 2 ellipses in icons)
        top_ellipses = EllipseDetection.filter_overlapping_ellipses(
            detected_ellipses, max_out=2)

        if len(top_ellipses) == 2:
            area1 = top_ellipses[0][1][0] * top_ellipses[0][1][1]
            area2 = top_ellipses[1][1][0] * top_ellipses[1][1][1]

            # If biggest ellipse is way larger than the second,
            # it's not wheels which is the only pair of circle acceptable
            # So we keep only the main one.
            if area1 > area2 * 3:
                return [top_ellipses[0]]

        return top_ellipses

    @staticmethod
    def filter_overlapping_ellipses(ellipses, overlap_threshold=0.6,
                                    max_out=2):
        """Filter overlapping ellipses, returning maximum 'max_out' results"""
        if not ellipses:
            return []

        filtered = []
        for candidate in ellipses:
            if len(filtered) >= max_out:
                break

            is_overlap = False
            c1 = candidate[0]
            # Radius based filtering
            r1 = max(candidate[1]) / 2

            for fixed in filtered:
                c2 = fixed[0]
                r2 = max(fixed[1]) / 2

                # distance between centers
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

                # if centers are closer than a % of the largest radius
                if dist < max(r1, r2) * overlap_threshold:
                    is_overlap = True
                    break

            if not is_overlap:
                filtered.append(candidate)

        return filtered


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

        detector = EllipseDetection()
        results = detector.detect_ellipses(binary)

        print(f"Found {len(results)} ellipse(s).")

        output_img = img.copy()
        for i, ell in enumerate(results):
            # Draw ellipses
            color = (255, 0, 0) if i == 0 else (0, 255, 0)
            cv2.ellipse(output_img, ell, color, 2)

        cv2.imshow("Detected Ellipses", output_img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
