# detector.py
import numpy as np
import supervision as sv
from ultralytics import YOLO
from config import MODEL_PATH, CONF_THRESHOLD, COCO_PERSON, COCO_SPORTS_BALL


class LacrosseDetector:
    """
    Wraps YOLOv8 detection + ByteTrack for persistent player IDs.

    For the test run we use a pretrained COCO model which detects
    "person" (class 0) as players. Replace MODEL_PATH in config.py
    with a lacrosse-fine-tuned model when available.
    """

    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model(
            frame,
            conf=CONF_THRESHOLD,
            classes=[COCO_PERSON, COCO_SPORTS_BALL],
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)
        return detections

    def split(self, detections: sv.Detections):
        """Return (player_detections, ball_detections)."""
        players = detections[detections.class_id == COCO_PERSON]
        ball    = detections[detections.class_id == COCO_SPORTS_BALL]
        return players, ball

    @staticmethod
    def foot_point(bbox: np.ndarray) -> np.ndarray:
        """Bottom-center of bounding box = player's foot position."""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, y2])

    @staticmethod
    def center_point(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
