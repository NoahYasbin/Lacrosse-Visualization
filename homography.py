# homography.py
from __future__ import annotations
import cv2
import numpy as np
import os


# Standard lacrosse keypoints in yards (x=width 0-60, y=length 0-110)
# These are the points you will click on the video frame in order.
FIELD_KEYPOINTS_YD = np.array([
    [0,    0],   # 1. Top-left corner
    [60,   0],   # 2. Top-right corner
    [0,  110],   # 3. Bottom-left corner
    [60, 110],   # 4. Bottom-right corner
    [0,   55],   # 5. Left midfield sideline
    [60,  55],   # 6. Right midfield sideline
    [0,   20],   # 7. Left restraining line (top)
    [60,  20],   # 8. Right restraining line (top)
], dtype=np.float32)

KEYPOINT_LABELS = [
    "1. Top-left corner",
    "2. Top-right corner",
    "3. Bottom-left corner",
    "4. Bottom-right corner",
    "5. Left midfield sideline",
    "6. Right midfield sideline",
    "7. Left restraining line (attack end)",
    "8. Right restraining line (attack end)",
]


class FieldHomography:
    def __init__(self):
        self.H: np.ndarray | None = None

    def compute(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> None:
        self.H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if self.H is None:
            raise ValueError("Homography failed — try picking clearer points.")
        print(f"Homography OK. Inliers: {mask.ravel().sum()}/{len(src_pts)}")

    def transform(self, points_px: np.ndarray) -> np.ndarray:
        """Map video pixel coords → field yard coords. Input shape (N,2)."""
        if self.H is None:
            raise RuntimeError("Call compute() first.")
        pts = points_px.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def save(self, path: str) -> None:
        np.save(path, self.H)
        print(f"Homography saved → {path}")

    def load(self, path: str) -> None:
        self.H = np.load(path)
        print(f"Homography loaded ← {path}")


def interactive_calibrate(frame: np.ndarray, save_path: str = "homography.npy") -> FieldHomography:
    """
    Show the first video frame in a matplotlib window and collect 8 click points.
    Uses matplotlib instead of cv2.imshow to avoid macOS Qt backend issues.
    """
    import matplotlib
    matplotlib.use("MacOSX")
    import matplotlib.pyplot as plt

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    clicks = []

    print("\n=== HOMOGRAPHY CALIBRATION ===")
    print("A window will open. Click these 8 points on the field IN ORDER:")
    for lbl in KEYPOINT_LABELS:
        print(f"  {lbl}")
    print("Close the window after click #8 to continue.\n")

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(rgb)
    ax.set_title(f"Click point 1/8:  {KEYPOINT_LABELS[0]}", fontsize=12)
    ax.axis("off")

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        if len(clicks) >= 8:
            return
        x, y = int(event.xdata), int(event.ydata)
        clicks.append([x, y])
        n = len(clicks)
        ax.plot(x, y, "o", color="cyan", markersize=10)
        ax.annotate(str(n), (x, y), color="yellow", fontsize=13,
                    fontweight="bold", ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")
        print(f"  [{n}/8] {KEYPOINT_LABELS[n-1]} → ({x}, {y})")
        if n < 8:
            ax.set_title(f"Click point {n+1}/8:  {KEYPOINT_LABELS[n]}", fontsize=12)
        else:
            ax.set_title("All 8 points collected — close this window to continue.", fontsize=12)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if len(clicks) < 8:
        raise RuntimeError(f"Only {len(clicks)}/8 points clicked — please redo calibration.")

    src_pts = np.array(clicks, dtype=np.float32)
    hom = FieldHomography()
    hom.compute(src_pts, FIELD_KEYPOINTS_YD)
    hom.save(save_path)
    return hom
