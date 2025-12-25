"""Module for subdividing images into smaller grid cells."""


class ImageSubdivision:
    """Subdivide images into grid cells"""

    @staticmethod
    def subdivide(image, rows, cols):
        """Divide image into rows x cols sub images"""
        sub_image_height = image.shape[0] // rows
        sub_image_width = image.shape[1] // cols

        sub_images = []

        for i in range(rows):
            for j in range(cols):
                x = j * sub_image_width
                y = i * sub_image_height
                sub_image = image[y:y + sub_image_height,
                                  x:x + sub_image_width].copy()
                sub_images.append(sub_image)

        return sub_images
