"""
Provides labelling logic for extracted squares.
"""

from typing import Dict, List, Tuple


def labelling(
    squares: Dict[int, List[Tuple]],
    file_terrminology: str
) -> Dict[str, List[Tuple]]:
    """
    Assign labels to grouped squares based on their Y-position order.

    Args:
        squares: Dictionary mapping Y-coordinates to lists of rectangles
        file_terminology: Terminology file identifier

    Returns:
        Dictionary mapping labels to their corresponding rectangles
    """
    labels = []
    if file_terrminology == "00000":
        labels = [
            'Warning',
            'Bomb',
            'Car',
            'Casualty',
            'Electricity',
            'Fire',
            'Fire_brigade'
        ]
    elif file_terrminology == "00001":
        labels = [
            'Gas',
            'Injury',
            'Paramedics',
            'Person',
            'Police',
            'Road_block',
            'Flood'
        ]
    else:
        print("Error: Unknown file terrminology")
        return {}

    # Sort groups by Y-coordinate (top to bottom)
    # Groups should already be sorted by Y-coordinate keys, this is a safeguard
    sorted_grps = sorted(squares.items(), key=lambda item: item[0])

    # Map labels to rectangle groups
    labeled_squares = {}
    for i, (_, rectangles) in enumerate(sorted_grps):
        if i < len(labels):
            labeled_squares[labels[i]] = rectangles
        else:
            print(f"Warning: {len(sorted_grps)} groups, {len(labels)} labels")
            break

    return labeled_squares
