# field_map.py
import cv2
import numpy as np
from config import FIELD_W_PX, FIELD_H_PX, FIELD_LENGTH_YD, PX_PER_YD

GRASS = (34, 139, 34)
WHITE = (255, 255, 255)


def yd_to_px(x_yd: float, y_yd: float) -> tuple:
    """Convert field yard coords to canvas pixel coords."""
    px = int(round(x_yd * PX_PER_YD))
    py = int(round(y_yd * PX_PER_YD))
    return px, py


def draw_lacrosse_field() -> np.ndarray:
    """
    Render a top-down lacrosse field diagram.
    Origin (0,0) = top-left. X = width (60 yd), Y = length (110 yd).
    """
    canvas = np.full((FIELD_H_PX, FIELD_W_PX, 3), GRASS, dtype=np.uint8)
    lw = 2

    # Alternating grass stripes (aesthetic)
    stripe_yd = 10
    for i in range(0, FIELD_LENGTH_YD, stripe_yd * 2):
        y0 = int(i * PX_PER_YD)
        y1 = int(min((i + stripe_yd) * PX_PER_YD, FIELD_H_PX))
        canvas[y0:y1, :] = (28, 120, 28)

    # Outer boundary
    cv2.rectangle(canvas, yd_to_px(0, 0), yd_to_px(60, 110), WHITE, lw)

    # Midfield line
    cv2.line(canvas, yd_to_px(0, 55), yd_to_px(60, 55), WHITE, lw)

    # Center circle (r = 10 yd)
    cx, cy = yd_to_px(30, 55)
    cv2.circle(canvas, (cx, cy), int(10 * PX_PER_YD), WHITE, lw)

    # Center dot
    cv2.circle(canvas, (cx, cy), 4, WHITE, -1)

    # Restraining lines (20 yd from each end line)
    cv2.line(canvas, yd_to_px(0, 20),  yd_to_px(60, 20),  WHITE, lw)
    cv2.line(canvas, yd_to_px(0, 90),  yd_to_px(60, 90),  WHITE, lw)

    # Wing lines (from restraining line corners to crease, both sides)
    for goal_line_y, restrain_y in [(15, 20), (95, 90)]:
        for side_x in [0, 60]:
            cv2.line(canvas, yd_to_px(side_x, restrain_y),
                     yd_to_px(30, goal_line_y), WHITE, 1)

    # Goals: 6 ft wide = 2 yd, centered at x=30
    for goal_y in [14, 94]:
        cv2.rectangle(canvas, yd_to_px(29, goal_y),
                      yd_to_px(31, goal_y + 2), WHITE, lw)

    # Goal lines (full width)
    cv2.line(canvas, yd_to_px(0, 15), yd_to_px(60, 15), WHITE, lw)
    cv2.line(canvas, yd_to_px(0, 95), yd_to_px(60, 95), WHITE, lw)

    # Creases (r = 9 ft ≈ 3 yd, centered on goal line)
    for crease_y in [15, 95]:
        cx2, cy2 = yd_to_px(30, crease_y)
        cv2.circle(canvas, (cx2, cy2), int(3 * PX_PER_YD), WHITE, lw)

    # Field label
    cv2.putText(canvas, "LACROSSE FIELD",
                yd_to_px(8, 54), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    return canvas
