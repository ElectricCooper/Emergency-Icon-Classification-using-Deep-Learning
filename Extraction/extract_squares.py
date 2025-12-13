"""
Square Detector module for detecting and processing squares in images
"""

from typing import List, Tuple, Dict
# pylint: disable=no-member
import cv2
import numpy as np


class SquareDetector:
    """Class for detecting and processing squares in images."""

    THRESH = 50  # For edge detection

    @staticmethod
    def angle(
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        pt0: Tuple[int, int]
    ) -> float:
        """
        Calculate cosine of angle between vectors from pt0->pt1 and pt0->pt2.

        Args:
            pt1: The first point
            pt2: The second point
            pt0: The origin point

        Returns:
            The cosine of the angle
        """
        dx1 = pt1[0] - pt0[0]
        dy1 = pt1[1] - pt0[1]
        dx2 = pt2[0] - pt0[0]
        dy2 = pt2[1] - pt0[1]

        dx1 = float(dx1)
        dy1 = float(dy1)
        dx2 = float(dx2)
        dy2 = float(dy2)
        return (dx1 * dx2 + dy1 * dy2) / np.sqrt(
            (dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10
        )

    @staticmethod
    def find_squares(image: np.ndarray) -> List[np.ndarray]:
        """
        Find squares in the given image.

        Args:
            image: The input image

        Returns:
            A list of detected squares (each square is a contour)
        """
        squares = []

        # Down-scale and up-scale the image to filter out noise
        down1 = cv2.pyrDown(image)
        down2 = cv2.pyrDown(down1)
        up1 = cv2.pyrUp(down2, dstsize=(down1.shape[1], down1.shape[0]))
        timg = cv2.pyrUp(up1, dstsize=(image.shape[1], image.shape[0]))

        # Search for squares in every color plane
        for c in range(3):
            # Extract the channel
            gray0 = timg[:, :, c]

            # Apply Canny edge detection
            gray = cv2.Canny(gray0, 0, SquareDetector.THRESH, apertureSize=3)
            gray = cv2.dilate(gray, None)

            # Find contours
            contours, _ = cv2.findContours(
                gray,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
                )

            # Test each contour
            for contour in contours:
                # Approximate the contour
                approx = cv2.approxPolyDP(
                    contour,
                    cv2.arcLength(contour, True) * 0.02,
                    True
                    )

                # Check if it is a square
                if (len(approx) == 4 and
                        abs(cv2.contourArea(approx)) > 1000 and
                        cv2.isContourConvex(approx)):

                    max_cosine = 0
                    for j in range(2, 5):
                        cosine = abs(SquareDetector.angle(
                            tuple(approx[j % 4][0]),
                            tuple(approx[j - 2][0]),
                            tuple(approx[j - 1][0])
                        ))
                        max_cosine = max(max_cosine, cosine)

                    # Accept if all angles are ~90 degrees
                    if max_cosine < 0.3:
                        squares.append(approx)

        return squares

    @staticmethod
    def map_squares(
        squares: List[np.ndarray],
        threshold: int = 10
    ) -> Dict[int, List[Tuple]]:
        """
        Map detected squares into groups based on their Y-coordinate.

        Args:
            image: The input image
            squares: The detected squares
            threshold: Y-coordinate threshold for grouping

        Returns:
            A dict in the form (Y-coordinate,list of rectangles (x, y, w, h)
        """
        grouped_squares = {}

        # Group squares by Y coordinate
        for square in squares:
            x, y, w, h = cv2.boundingRect(square)

            added = False
            for group_y in list(grouped_squares.keys()):
                if abs(group_y - y) <= threshold:
                    grouped_squares[group_y].append((x, y, w, h))
                    added = True
                    break

            if not added:
                grouped_squares[y] = [(x, y, w, h)]

        for group_y in grouped_squares:
            grouped_squares[group_y] = sorted(
                                        grouped_squares[group_y],
                                        key=lambda r: r[0]
                                        )

        return grouped_squares

    @staticmethod
    def filter_squares(
        squares: List[np.ndarray],
        area_min: int,
        area_max: int,
        tolerance: int = 20
    ) -> List[np.ndarray]:
        """
        Filter out overlapping or similar squares based on area and tolerance.

        Args:
            squares: The detected squares
            area_min: Minimum area size
            area_max: Maximum area size
            tolerance: Tolerance for determining similar squares

        Returns:
            Filtered list of squares
        """
        filtered_squares = []
        keep = [True] * len(squares)

        # Remove duplicate / similar squares
        for i, sq1 in enumerate(squares):
            if not keep[i]:
                continue

            x1, y1, w1, h1 = cv2.boundingRect(sq1)
            rect1_br_x = x1 + w1
            rect1_br_y = y1 + h1

            for j, sq2 in enumerate(squares[i + 1:], start=i + 1):
                if not keep[j]:
                    continue

                x2, y2, w2, h2 = cv2.boundingRect(sq2)
                rect2_br_x = x2 + w2
                rect2_br_y = y2 + h2

                is_similar = (
                    abs(x1 - x2) < tolerance and
                    abs(y1 - y2) < tolerance and
                    abs(rect1_br_x - rect2_br_x) < tolerance and
                    abs(rect1_br_y - rect2_br_y) < tolerance
                )

                if is_similar:
                    area1 = w1 * h1
                    area2 = w2 * h2

                    if area1 < area2:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break   # stop comparing

        # Filter by area
        for i, sq in enumerate(squares):
            if not keep[i]:
                continue

            _, _, w, h = cv2.boundingRect(sq)
            area = w * h

            longer_side = max(w, h)
            shorter_side = min(w, h)

            if longer_side > 2 * shorter_side:  # reject non-square rectangles
                continue

            if area_min < area < area_max:
                filtered_squares.append(sq)

        return filtered_squares
