""" Feature extraction pipeline"""

import sys
from pathlib import Path
import json
# pylint: disable=no-member
import cv2

from geometric_features import GeometricFeatures
from moments import MomentsFeatures
from poi import POIDetection
from lines import LineDetection
from ellipse import EllipseDetection
from subdivider import ImageSubdivision
from diagonal import DiagonalAnalysis


class IconFeatureExtractor:
    """Consolidated feature extraction pipeline with optimized logic"""

    def __init__(self, rows=2, cols=3):
        self.rows = rows
        self.cols = cols

    def extract_core_features(self, binary):
        """Helper to extract shared features for both global and sub-zones"""
        # Lines
        filtered_lines = LineDetection.detect_lines(binary)
        line_directions = LineDetection.classify_directions(filtered_lines)

        # POI
        total_corners, _ = POIDetection.detect_all_corners(binary)
        sharp_corners, _ = POIDetection.detect_sharp_corners(binary)

        # Geometry & Moments
        perimeter = GeometricFeatures.calculate_perimeter(binary)
        area = GeometricFeatures.compute_area(binary)
        compactness = GeometricFeatures.compute_compactness(perimeter, area)
        hu_moments = MomentsFeatures.get_hu_moments(binary)

        return {
            'perimeter': perimeter,
            'area': int(area),
            'compactness': compactness,
            'hu_moments': hu_moments,
            'corners_count': total_corners,
            'sharp_corners_count': sharp_corners,
            'line_directions': line_directions
        }

    def extract_features_from_image(self, image_path):
        """Extract features from a single image using consolidated logic"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # Subdivision features extraction
        sub_zones = ImageSubdivision.subdivide(binary, self.rows, self.cols)
        subdivision_features = []
        for zone in sub_zones:
            subdivision_features.append(self.extract_core_features(zone))

        # Global features extraction
        features = {'subdivisions': subdivision_features}
        global_f = self.extract_core_features(binary)

        # Adding global features
        ellipses = EllipseDetection.detect_ellipses(binary)
        diag = DiagonalAnalysis.analyze_diagonal(binary)
        convex_area = GeometricFeatures.calculate_convex_area(binary)
        avg_centr_radius = MomentsFeatures.average_centroidal_radius(binary)

        global_f.update({
            'ellipse_count': len(ellipses),
            'diagonal_length': diag['length'],
            'diagonal_angle': diag['angle'],
            'convex_area': convex_area,
            'avg_centroidal_radius': avg_centr_radius
        })

        features['global'] = global_f
        return features

    def process_dataset(self, data_dir, output_json='features_dataset.json'):
        """Walks through folders, extracts features, and save to JSON"""
        data_dir = Path(data_dir)
        dataset = []

        for label_folder in filter(Path.is_dir, data_dir.iterdir()):
            label = label_folder.name
            for img_file in label_folder.glob('*.png'):
                feat = self.extract_features_from_image(img_file)
                if feat:
                    feat.update({'label': label, 'filename': img_file.name})
                    dataset.append(feat)

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path>")
        sys.exit(1)

    folder_path = Path(sys.argv[1])
    IconFeatureExtractor().process_dataset(folder_path)