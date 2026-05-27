"""
Image Processing — Grid Converter
----------------------------------
Converts real-world floor plan / maze images into binary grids
for AI algorithm input.

Pipeline:
  1. Load image (BGR via OpenCV)
  2. Convert to grayscale
  3. Adaptive thresholding (handles uneven lighting)
  4. Morphological cleanup (remove noise)
  5. Downscale to manageable grid resolution
  6. Output: binary grid (1=walkable, 0=obstacle)
"""

import cv2
import numpy as np


class ImageToGrid:
    """
    Converts an image file to a binary navigable grid.
    Supports multiple environments: floorplans, mazes, street maps.
    """

    def __init__(self, image_path, grid_scale=1.0, white_thresh=200):
        """
        grid_scale: fraction to downscale (1.0 = full resolution)
        white_thresh: pixel brightness >= this → walkable
        """
        self.image_path = image_path
        self.grid_scale = grid_scale
        self.white_thresh = white_thresh

    def load(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {self.image_path}")
        return img

    def preprocess(self, img):
        """
        Returns binary image: 255 = walkable, 0 = obstacle.
        Uses adaptive threshold for robustness to lighting variation.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Optional downscale for large images
        if self.grid_scale != 1.0:
            h = int(gray.shape[0] * self.grid_scale)
            w = int(gray.shape[1] * self.grid_scale)
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

        # Simple threshold: white pixels = walkable
        _, binary = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY)

        # Morphological opening: remove isolated noise pixels
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Block borders (prevent agents escaping grid edge)
        h, w = binary.shape
        border = 3
        binary[:border, :] = 0
        binary[h - border:, :] = 0
        binary[:, :border] = 0
        binary[:, w - border:] = 0

        return gray, binary

    def to_grid(self, binary):
        """Convert binary image to integer grid: 1=walkable, 0=obstacle."""
        return (binary > 0).astype(int).tolist()

    def process(self):
        """
        Full pipeline.
        Returns: (grid, binary_img, original_img)
        """
        img = self.load()
        gray, binary = self.preprocess(img)
        grid = self.to_grid(binary)
        return grid, binary, img, gray


def grid_to_numpy(grid):
    """Helper: convert list-of-lists to numpy array."""
    return np.array(grid, dtype=np.int8)
