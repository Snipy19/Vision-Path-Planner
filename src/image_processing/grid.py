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

    def __init__(self, image_path, grid_scale=1.0, white_thresh=200,
                 max_dimension=None, border=3, mode="auto"):
        """
        grid_scale: fraction to downscale (1.0 = full resolution)
        white_thresh: pixel brightness >= this → walkable
        max_dimension: optional safety cap for interactive/search workloads
        border: number of obstacle cells to reserve around the image edge
        mode: auto, simple, 3d, map, or occupancy
        """
        self.image_path = image_path
        self.grid_scale = grid_scale
        self.white_thresh = white_thresh
        self.max_dimension = max_dimension
        self.border = max(0, int(border))
        self.mode = mode.lower().strip()

    def load(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {self.image_path}")
        return img

    def preprocess(self, img, mode=None):
        """
        Returns binary image: 255 = walkable, 0 = obstacle.
        Uses adaptive threshold for robustness to lighting variation.
        """
        # Optional downscale for large images.  A cap prevents accidental
        # quadratic search costs when a camera image is several thousand px.
        scale = float(self.grid_scale)
        if scale <= 0:
            raise ValueError("grid_scale must be greater than zero")
        if self.max_dimension:
            scale = min(scale, float(self.max_dimension) / max(img.shape[:2]))
        work_img = img
        if scale < 1.0:
            h = max(1, int(round(img.shape[0] * scale)))
            w = max(1, int(round(img.shape[1] * scale)))
            work_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)

        selected_mode = (mode or self.mode).lower().strip()
        if selected_mode == "auto":
            hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
            mean_saturation = float(np.mean(hsv[:, :, 1]))
            selected_mode = "3d" if mean_saturation >= 32.0 else "simple"
        if selected_mode in {"3d", "complex", "render"}:
            # Rendered rooms usually have a bright, low-saturation floor and
            # colorful/dark furniture.  HSV separates those properties more
            # reliably than a grayscale threshold.
            hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            floor = ((saturation < 85) & (value > 125)) | \
                    ((saturation < 45) & (value > 95))
            binary = (floor.astype(np.uint8) * 255)
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        elif selected_mode in {"occupancy", "map", "simple", "bw"}:
            _, binary = cv2.threshold(gray, self.white_thresh, 255,
                                      cv2.THRESH_BINARY)
        else:
            raise ValueError(
                f"Unknown image mode '{selected_mode}'. Use auto, simple, 3d, map, or occupancy.")

        # Morphological opening: remove isolated noise pixels
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Block borders (prevent agents escaping grid edge)
        h, w = binary.shape
        border = min(self.border, max(0, min(h, w) // 2))
        if border:
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
        # Keep the rendered image in the same coordinate system as the grid.
        # This avoids overlays being shifted when max_dimension/grid_scale is used.
        if img.shape[:2] != gray.shape[:2]:
            img = cv2.resize(img, (gray.shape[1], gray.shape[0]),
                             interpolation=cv2.INTER_AREA)
        return grid, binary, img, gray


def grid_to_numpy(grid):
    """Helper: convert list-of-lists to numpy array."""
    return np.array(grid, dtype=np.int8)
