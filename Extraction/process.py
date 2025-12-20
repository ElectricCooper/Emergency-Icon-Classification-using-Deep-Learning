"""
Script to process cropped squares
"""

# pylint: disable=no-member
import cv2
import numpy as np


def process_square(
    cropped_square: np.ndarray,
    padding: int = 20
) -> np.ndarray:
    """
    Detect the drawing within a cropped square, zoom to its bounding box,
    and return a binary image resized to the original dimensions.

    Args:
        cropped_square: Input image.
        padding: Padding (in pixels) added around the detected icon.

    Returns:
        Binary image, zoomed on the icon, with original dimensions.
    """
    height, width = cropped_square.shape[:2]

    gray = cv2.cvtColor(cropped_square, cv2.COLOR_BGR2GRAY)

    # Binarize for contour detection
    # OTSU is used to automatically determine threshold value
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Bounding box over contours
    points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(points)

    # Expand bounding box with padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(width - x, w + 2 * padding)
    h = min(height - y, h + 2 * padding)

    # Crop and resize binary image
    icon = binary[y:y + h, x:x + w]
    icon_resized = cv2.resize(
                    icon,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

    # Morphological opening to clean noise
    # might remove depending on behavior with dataset
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_icon = cv2.morphologyEx(icon_resized, cv2.MORPH_OPEN, kernel)
    # if too much is removed, return previous image
    if cv2.countNonZero(clean_icon) < 0.8 * cv2.countNonZero(icon_resized):
        return icon_resized

    return clean_icon
