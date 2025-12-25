"""Ellipse detection module"""

import numpy as np
# pylint: disable=no-member
import cv2


class EllipseDetection:
    """Detect ellipses in images"""

    @staticmethod
    def detect_ellipses(image, min_ratio=0.3, min_area=200):
        """Detect ellipses in an image"""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        detected_ellipses = []
        for contour in contours:
            # An ellipse needs at least 5 points
            if len(contour) >= 5:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    ellipse = cv2.fitEllipse(contour)
                    # Aspect ratio
                    w, h = ellipse[1]
                    ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0

                    # Circularity (making sure it's not a random pt scatter)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter == 0:
                        circularity = (4*np.pi*area)/(perimeter**2)
                    else:
                        circularity = 0

                    if ratio > min_ratio and circularity > 0.4:  # Thresholds
                        detected_ellipses.append(ellipse)

        # Sort (descending) by area
        detected_ellipses.sort(key=lambda e: e[1][0] * e[1][1], reverse=True)
        # Filter overlapping ellipses and select top 2
        # (we don't expect more than 2 ellipses in icons)
        return EllipseDetection.filter_overlapping_ellipses(detected_ellipses,
                                                            max_out=2)

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
