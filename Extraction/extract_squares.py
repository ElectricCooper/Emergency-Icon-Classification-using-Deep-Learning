"""
Square Detector module for detecting and processing squares in images
"""

from typing import List, Tuple, Dict
# pylint: disable=no-member
import cv2
import numpy as np


class SquareDetector:
    """Class for detecting and processing squares in images."""

    THRESH = 10  # For edge detection
    AREA_THRESHOLD = 8000  # Minimum area to consider
    EXPECTED_COLS = 5
    EXPECTED_ROWS = 7
    EXPECTED_TOTAL = 35  # Expected number of icon squares

    @staticmethod
    def filter_by_aspect_ratio(
        squares: List[np.ndarray],
        max_ratio: float = 2.0
    ) -> List[np.ndarray]:
        """
        Filter out rectangles that are too long compared to their width.

        Args:
            squares: Detected squares
            max_ratio: Maximum allowed aspect ratio

        Returns:
            Filtered list of squares
        """
        filtered = []

        for sq in squares:
            _, _, w, h = cv2.boundingRect(sq)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio <= max_ratio:
                filtered.append(sq)
        return filtered

    @staticmethod
    def find_squares(image: np.ndarray):
        """
        Find large blobs in the given image.

        As the forms mostly contain squares and rectangles,
        most of detected blobs are squares.

        We also filter the rectangles to give back only squares.
        (Aspect ratio filterring can also filter some other abnormal blobs)

        Args:
            image: The input image

        Returns:
            A list of detected squares (each square is a contour)
        """
        squares = []

        # Search for squares
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        gray = cv2.Canny(gray, 0, 0)
        gray = cv2.dilate(gray, None)

        # Find contours
        contours, _ = cv2.findContours(
            gray,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
            )

        # Testing each contour
        for contour in contours:

            # Skip small areas (noise)
            # Verification moved here to avoid processing small contours
            area = abs(cv2.contourArea(contour))
            if area < SquareDetector.AREA_THRESHOLD:
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            squares.append(box)

        # Filter rectangles out
        squares = SquareDetector.filter_by_aspect_ratio(squares)
        return squares

    @staticmethod
    def row_map_by_area(
        squares: List[np.ndarray]
    ) -> Dict[int, List[Tuple]]:
        """
        Select 35 squares by area similarity,
        then group them into rows using Y-ordering,

        Returns:
            Dict[row_index]: list of rectangles for each row
        """
        if len(squares) < SquareDetector.EXPECTED_TOTAL:
            raise ValueError(
                "Not enough squares: expected at least "
                + f"{SquareDetector.EXPECTED_TOTAL}, got {len(squares)}"
            )

        # Selecting 35 squares by area similarity
        rects = [(sq, cv2.boundingRect(sq)) for sq in squares]
        areas = np.array([w * h for _, (_, _, w, h) in rects])

        median_area = np.median(areas)
        diffs = np.abs(areas - median_area)

        selected_idx = np.argsort(diffs)[:SquareDetector.EXPECTED_TOTAL]
        selected_rects = [rects[i][1] for i in selected_idx]

        # Sort by y center (top to bottom)
        selected_rects.sort(key=lambda r: r[1] + r[3] / 2)

        # Make rows
        grid: Dict[int, List[Tuple]] = {}

        for row_idx in range(SquareDetector.EXPECTED_ROWS):
            start = row_idx * SquareDetector.EXPECTED_COLS
            end = start + SquareDetector.EXPECTED_COLS

            row = selected_rects[start:end]

            if len(row) != SquareDetector.EXPECTED_COLS:
                raise RuntimeError(
                    f"Row {row_idx} has {len(row)} squares" +
                    f"instead of {SquareDetector.EXPECTED_COLS}"
                )

            grid[row_idx] = row

        return grid

    @staticmethod
    def remove_duplicates(
            squares: List[np.ndarray],
            tolerance: int = 30
    ) -> List[np.ndarray]:
        """
        Remove overlapping or nearly identical squares.

        Args:
            squares: List of detected squares
            tolerance: Distance tolerance for considering squares as duplicates

        Returns:
            List of squares without duplicates
        """
        keep = [True] * len(squares)

        for i, sq1 in enumerate(squares):
            if not keep[i]:
                continue

            x1, y1, w1, h1 = cv2.boundingRect(sq1)
            area1 = w1 * h1

            for j in range(i + 1, len(squares)):
                if not keep[j]:
                    continue

                x2, y2, w2, h2 = cv2.boundingRect(squares[j])
                area2 = w2 * h2

                # Check if squares are similar in position
                is_similar = (
                    abs(x1 - x2) < tolerance and
                    abs(y1 - y2) < tolerance and
                    abs((x1 + w1) - (x2 + w2)) < tolerance and
                    abs((y1 + h1) - (y2 + h2)) < tolerance
                )

                if is_similar:
                    # Keep the bigger one
                    if area1 > area2:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        filtered = [sq for i, sq in enumerate(squares) if keep[i]]
        return filtered
