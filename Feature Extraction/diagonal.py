""" Module for analyzing diagonal properties of drawings """

import sys
from pathlib import Path
import numpy as np
# pylint: disable=no-member
import cv2


class DiagonalAnalysis:
    """Analyze diagonal properties of drawings"""

    @staticmethod
    def analyze_diagonal(image):
        """Find and measure diagonal of the drawing"""
        # Safe checking image is 1-channel (binary from process.py)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find coordinates of all non-zero (white) pixels
        coords = cv2.findNonZero(image)

        if coords is None:
            return {'length': 0, 'angle': 0}

        _, _, w, h = cv2.boundingRect(coords)  # Bounding box

        # Calculate diagonal length and angle
        diagonal_length = np.sqrt(w**2 + h**2)
        diagonal_angle = np.degrees(np.arctan2(h, w))

        return {
            'length': round(diagonal_length, 3),
            'angle': round(diagonal_angle, 3)
        }


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

        analysis = DiagonalAnalysis().analyze_diagonal(binary)

        print(f"Diagonal Length: {analysis['length']}")
        print(f"Diagonal Angle: {analysis['angle']} degrees")

        output_img = img.copy()
        coordinates = cv2.findNonZero(binary)
        if coordinates is not None:
            x, y, width, height = cv2.boundingRect(coordinates)
            pt1 = (x, y)
            pt2 = (x + width, y + height)
            # Bounding box
            cv2.rectangle(output_img, pt1, pt2, (0, 255, 0), 2)
            # Diagonal line
            cv2.line(output_img, pt1, pt2, (255, 0, 0), 2)

            cv2.imshow("Diagonal Analysis", output_img)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
