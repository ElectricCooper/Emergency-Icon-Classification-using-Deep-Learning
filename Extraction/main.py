"""
Main script for processing images and extracting labeled squares.
Usage: python3 main.py <source_directory>
"""

import sys
from pathlib import Path
# pylint: disable=no-member
import cv2
from extract_squares import SquareDetector
from label import labelling


def extract_terminology(filename: str) -> str:
    """
    Extract terminology code from filename.

    Args:
        filename: Name of file

    Returns:
        Terminology code ("00000" or "00001") or empty string if not found
    """
    if filename.startswith("00000"):
        return "00000"
    elif filename.startswith("00001"):
        return "00001"
    else:
        return ""


def main():
    """Main"""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <source_directory>")
        sys.exit(1)

    source_path = Path(sys.argv[1])
    extracted_path = source_path.parent / "extracted"
    extracted_path.mkdir(exist_ok=True)

    # Parameters
    area_min = 50000
    area_max = 100000
    tolerance = 25

    # Statistics
    total_processed = 0
    label_counters = {}  # Track counter for each label

    # Process all PNG files in the source directory
    png_files = sorted(list(source_path.glob("*.png")))

    if not png_files:
        print(f"No PNG files found in {source_path}")
        sys.exit(1)

    for file_path in png_files:
        filename = file_path.name
        print(f"Processing: {filename}")

        # Extract terminology from filename
        terminology = extract_terminology(filename)
        if not terminology:
            print("Rejected - filename must start with '00000' or '00001'")
            continue

        # Read image
        img = cv2.imread(str(file_path))
        if img is None:
            print("Failed to read image")
            continue

        # Detect all squares
        squares = SquareDetector.find_squares(img)

        # Filter squares
        filtered_squares = SquareDetector.filter_squares(
            squares, area_min, area_max, tolerance
        )

        if not filtered_squares:
            print("Rejected - no squares after filtering")
            continue

        # Map squares to groups by Y-coordinate
        grouped_squares = SquareDetector.map_squares(filtered_squares)

        # Assign labels to groups
        labeled_squares = labelling(grouped_squares, terminology)

        if not labeled_squares:
            print("Failed to assign labels")
            continue

        # Process each labeled group
        squares_extracted = 0
        for label, rectangles in labeled_squares.items():
            # Create label folder if it doesn't exist
            label_folder = extracted_path / label
            label_folder.mkdir(exist_ok=True)

            # Initialize counter for this label if needed
            if label not in label_counters:
                label_counters[label] = 0

            # Save each square in the group
            for rect in rectangles:
                x, y, w, h = rect
                cropped_square = img[y:y+h, x:x+w]

                # Generate filename with counter
                counter = label_counters[label]
                image_filename = f"{label}_{counter:05d}.png"
                image_path = label_folder / image_filename

                cv2.imwrite(str(image_path), cropped_square)

                label_counters[label] += 1
                squares_extracted += 1

        total_processed += 1
        print(f"Extracted {squares_extracted} squares")

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Total files processed: {total_processed}")
    print("\nSquares extracted by label:")
    for label, count in sorted(label_counters.items()):
        print(f"  {label:20s}: {count:5d} squares")

    print(f"\nResults saved to: {extracted_path}")


if __name__ == "__main__":
    main()
