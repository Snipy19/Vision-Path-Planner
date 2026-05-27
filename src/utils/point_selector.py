"""
Interactive Point Selector
---------------------------
Click on image to select Start (first click) and Goal (second click).
Falls back to auto-selection if display not available.
"""

import cv2
import numpy as np
import random


_clicks = []


def _mouse_callback(event, x, y, flags, param):
    global _clicks
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicks) < 2:
        _clicks.append((y, x))   # (row, col)
        print(f"  Point {len(_clicks)} selected: row={y}, col={x}")


def select_points(image, window_title="Click: Start (1st), Goal (2nd)"):
    """
    Opens an OpenCV window. User clicks Start, then Goal.
    Returns: (start, goal) as (row, col) tuples.
    Falls back to random walkable points if display unavailable.
    """
    global _clicks
    _clicks = []

    try:
        display = image.copy()
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, min(1000, image.shape[1]),
                         min(700, image.shape[0]))
        cv2.setMouseCallback(window_title, _mouse_callback)

        print("\n[POINT SELECTOR] Click START point, then GOAL point on the image.")
        print("Press ENTER when done, ESC to cancel.\n")

        while True:
            disp = display.copy()
            for i, pt in enumerate(_clicks):
                color = (0, 0, 255) if i == 0 else (0, 255, 0)
                label = "S" if i == 0 else "G"
                cv2.circle(disp, (pt[1], pt[0]), 7, color, -1)
                cv2.putText(disp, label, (pt[1]+8, pt[0]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(window_title, disp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(_clicks) == 2:   # ENTER
                break
            if key == 27:   # ESC
                _clicks = []
                break

        cv2.destroyAllWindows()

    except Exception:
        _clicks = []

    if len(_clicks) == 2:
        return _clicks[0], _clicks[1]

    # Fallback: auto-select
    print("[INFO] Using auto-selected start and goal points.")
    return None, None   # Caller should handle


def auto_select_points(grid, seed=42):
    """
    Automatically select start and goal from walkable cells.
    Tries to pick points far apart (top-left region vs bottom-right).
    """
    random.seed(seed)
    rows, cols = len(grid), len(grid[0])
    walkable = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]

    if not walkable:
        raise ValueError("No walkable cells found in grid.")

    # Quadrant-based selection for distance
    q1 = [(r, c) for (r, c) in walkable if r < rows // 2 and c < cols // 2]
    q4 = [(r, c) for (r, c) in walkable if r >= rows // 2 and c >= cols // 2]

    start = random.choice(q1) if q1 else walkable[0]
    goal  = random.choice(q4) if q4 else walkable[-1]

    return start, goal


def fix_to_walkable(grid, pt, max_search=15):
    """
    If pt is on an obstacle, find nearest walkable cell.
    """
    r, c = pt
    rows, cols = len(grid), len(grid[0])

    if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 1:
        return pt

    for radius in range(1, max_search + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1):
                    return (nr, nc)
    return pt
