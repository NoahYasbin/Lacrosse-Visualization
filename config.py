# config.py

# Men's lacrosse field dimensions (yards)
FIELD_WIDTH_YD  = 60
FIELD_LENGTH_YD = 110

# Visualization canvas (pixels) — portrait orientation
FIELD_W_PX = 600
FIELD_H_PX = 1100

# Scale
PX_PER_YD = FIELD_W_PX / FIELD_WIDTH_YD  # 10 px/yd

# Pressure distance thresholds (yards on field)
PRESSURE_GREEN  = 8.0   # > 8 yd  → free (green)
PRESSURE_YELLOW = 4.0   # 4–8 yd  → contested (yellow)
#                         < 4 yd  → marked (red)

# Player circle radius on minimap (px)
PLAYER_RADIUS_PX = 14

# YOLO model — uses pretrained COCO model for test run (detects "person")
# Replace with a lacrosse-fine-tuned model path when available
MODEL_PATH = "yolov8n.pt"

# Confidence threshold for detections
CONF_THRESHOLD = 0.4

# COCO class IDs (used with pretrained model)
COCO_PERSON      = 0
COCO_SPORTS_BALL = 32

# Minimap scale when overlaid on video frame (picture-in-picture)
MINIMAP_SCALE = 0.22

# How often to re-fit the team color classifier (every N frames)
TEAM_REFIT_INTERVAL = 60
