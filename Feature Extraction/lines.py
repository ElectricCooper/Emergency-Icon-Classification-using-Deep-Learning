"""Line detection module"""

import sys
from pathlib import Path
import numpy as np
# pylint: disable=no-member
import cv2


class LineDetection:
    """Detect and regroup lines in images"""

    @staticmethod
    def detect_lines(image):
        """Detect lines using Hough Transform"""
        # Apply Canny edge detection
        lines = cv2.HoughLinesP(
            image,
            rho=1,
            theta=np.pi/180,
            threshold=20,      # min intersections to detect line
            minLineLength=15,  # min length of line
            maxLineGap=5       # max gap between pixels
        )

        if lines is None:
            return []

        raw_lines = [line[0] for line in lines]

        # return list of x1,y1,x2,y2
        return LineDetection.group_lines_with_limits(raw_lines)

    @staticmethod
    def classify_directions(lines):
        """Classify lines into horizontal, vertical, and diagonal"""
        results = {'horizontal': 0, 'vertical': 0, 'diag1': 0, 'diag2': 0}

        for x1, y1, x2, y2 in lines:
            label, _ = LineDetection.get_line_info(x1, y1, x2, y2)
            results[label] += 1

        return results

    @staticmethod
    def get_line_info(x1, y1, x2, y2):
        """Calculate angle and return label with color"""
        dx = x2 - x1
        dy = y2 - y1
        # Calculate angle in degrees (0-180)
        angle = np.degrees(np.arctan2(dy, dx)) % 180  # Normalize to 0-180

        if angle <= 22.5 or angle >= 157.5:
            return 'horizontal', (255, 0, 0)  # Blue
        elif 67.5 <= angle <= 112.5:
            return 'vertical', (0, 255, 0)  # Green
        elif 22.5 < angle < 67.5:
            return 'diag1', (0, 0, 255)  # Red
        else:
            return 'diag2', (0, 255, 255)  # Yellow

    @staticmethod
    def group_lines_with_limits(lines):
        """Filters lines with spatial separation and quantity limits"""
        categories = {'horizontal': [],
                      'vertical': [],
                      'diag1': [],
                      'diag2': []}

        for line in lines:
            x1, y1, x2, y2 = line
            # Calculate length for sorting
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            label, _ = LineDetection.get_line_info(x1, y1, x2, y2)

            if label in categories:
                categories[label].append((line, length))

        final_lines = []

        # Limit horizontal lines to 3
        categories['horizontal'].sort(key=lambda x: x[1], reverse=True)
        h_kept = []
        for line, _ in categories['horizontal']:
            if len(h_kept) < 3:
                mid_y = (line[1] + line[3]) / 2
                # Keep if vertical distance from others is > 10px
                if all(abs(mid_y - ((h[1] + h[3]) / 2)) > 10 for h in h_kept):
                    h_kept.append(line)

        # Limit vertical lines to 3
        categories['vertical'].sort(key=lambda x: x[1], reverse=True)
        v_kept = []
        for line, _ in categories['vertical']:
            if len(v_kept) < 3:
                mid_x = (line[0] + line[2]) / 2
                # Keep if horizontal distance from others is > 10px
                if all(abs(mid_x - ((v[0] + v[2]) / 2)) > 10 for v in v_kept):
                    v_kept.append(line)

        # Filtering diagonals too close to each other (no limits)
        for d_type in ['diag1', 'diag2']:
            categories[d_type].sort(key=lambda x: x[1], reverse=True)
            d_kept = []

            for line, _ in categories[d_type]:
                # Center point
                mid_x = (line[0] + line[2]) / 2
                mid_y = (line[1] + line[3]) / 2

                # Check distance
                is_too_close = False
                for k_line in d_kept:
                    k_mid_x = (k_line[0] + k_line[2]) / 2
                    k_mid_y = (k_line[1] + k_line[3]) / 2
                    dist = np.sqrt((mid_x - k_mid_x)**2 + (mid_y - k_mid_y)**2)

                    if dist < 5:
                        is_too_close = True
                        break

                if not is_too_close:
                    d_kept.append(line)

        final_lines.extend(h_kept)
        final_lines.extend(v_kept)
        final_lines.extend(d_kept)

        return final_lines


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

        detector = LineDetection()

        detected_lines = detector.detect_lines(binary)
        directions = detector.classify_directions(detected_lines)

        print(f"Detected lines: {len(detected_lines)}")
        print("Line directions:", directions)

        output_img = img.copy()
        for x_1, y_1, x_2, y_2 in detected_lines:
            _, color = detector.get_line_info(x_1, y_1, x_2, y_2)
            cv2.line(output_img, (x_1, y_1), (x_2, y_2), color, 2)

        cv2.imshow("Detected Lines", output_img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
