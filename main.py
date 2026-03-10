# main.py
import cv2
import numpy as np
import argparse
import os
import sys
import json
from scipy.spatial import KDTree

from detector import LacrosseDetector
from team_classifier import TeamClassifier, REF

# Pressure distance thresholds (pixels in the video frame)
PRESSURE_GREEN_PX  = 60   # > 60px  → free (green)
PRESSURE_YELLOW_PX = 120  # 120-60px → contested (yellow)
#                           < 30px  → marked (red)

TEAM_REFIT_INTERVAL = 60

# Team A / Team B circle fill colors (used for defense & unclassified offense)
TEAM_COLORS = {
    0: (220, 80,  40),    # Team A — blue/teal
    1: (40,  80, 220),    # Team B — red/orange
    2: (80,  80,  80),    # Refs   — dark grey
}


def load_player_names(path: str) -> dict:
    """Load players.json → {track_id_str: display_name}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_label(tid: int, player_names: dict) -> str:
    return player_names.get(str(tid), f"#{tid}")


def pressure_color(dist_px: float) -> tuple:
    if dist_px >= PRESSURE_YELLOW_PX:   # >= 120px → free (green)
        return (0, 210, 0)
    if dist_px >= PRESSURE_GREEN_PX:    # 60-120px → contested (light yellow)
        return (0, 230, 255)
    return (0, 0, 210)                  # < 60px → marked (red)


def draw_players(frame, bboxes, track_ids, teams, offense_team, player_names):
    out = frame.copy()

    # Only actual defenders (not refs) count for pressure
    R = 14
    def_centers = []
    for i, team in enumerate(teams):
        if team != offense_team and team != REF:
            x1, y1, x2, y2 = bboxes[i].astype(int)
            def_centers.append([(x1 + x2) / 2, y2 - R])   # foot position

    tree = KDTree(def_centers) if def_centers else None

    R = 14   # fixed radius — same size for every player every frame

    for i, (bbox, tid, team) in enumerate(zip(bboxes, track_ids, teams)):
        x1, y1, x2, y2 = bbox.astype(int)
        cx = (x1 + x2) // 2
        # Place circle at feet: center it so bottom edge sits on y2
        fy = y2 - R

        if team == REF:
            pts = np.array([
                [cx,     fy - R],
                [cx + R, fy    ],
                [cx,     fy + R],
                [cx - R, fy    ],
            ], np.int32)
            cv2.polylines(out, [pts], True, (200, 200, 200), 2)
            cv2.putText(out, "REF", (cx - 10, fy + R + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            continue

        if team == offense_team:
            color = pressure_color(float(tree.query([cx, fy])[0])) if tree else (0, 210, 0)
        else:
            color = TEAM_COLORS.get(int(team), (150, 150, 150))

        cv2.circle(out, (cx, fy), R, color, -1)
        cv2.circle(out, (cx, fy), R, (255, 255, 255), 2)

        label = get_label(int(tid), player_names)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(out, label, (cx - tw // 2, fy + R + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def draw_legend(frame):
    items = [
        ((0, 210, 0),   "Offense - free (>120px)"),
        ((0, 230, 255), "Offense - contested (60-120px)"),
        ((0, 0,   210), "Offense - marked (<60px)"),
        ((220, 80, 40), "Defense"),
        ((200, 200, 200), "Referee"),
    ]
    x, y = 15, 20
    for color, label in items:
        cv2.circle(frame, (x + 8, y), 8, color, -1)
        cv2.circle(frame, (x + 8, y), 8, (255, 255, 255), 1)
        cv2.putText(frame, label, (x + 22, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return frame


def save_id_reference(frame, bboxes, track_ids, path: str):
    """Save a labeled screenshot so you can see which tracker ID = which player."""
    ref = frame.copy()
    for bbox, tid in zip(bboxes, track_ids):
        x1, y1, x2, y2 = bbox.astype(int)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.rectangle(ref, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"ID:{tid}"
        cv2.putText(ref, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, ref)
    print(f"ID reference image saved → {path}")
    print("Open it to see which tracker ID belongs to which player,")
    print("then fill in players.json accordingly.\n")


def make_writer(path: str, fps: float, fw: int, fh: int):
    """Try avc1 (H.264) first — best for macOS QuickTime. Fall back to mp4v."""
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (fw, fh))
        if w.isOpened():
            print(f"Video codec: {codec}")
            return w
    print("ERROR: Could not open VideoWriter. Check output path.")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Lacrosse pressure tracker.")
    p.add_argument("--video",        required=True)
    p.add_argument("--output",       default="output.mp4")
    p.add_argument("--offense",      type=int, default=0,
                   help="Team ID (0 or 1) on offense")
    p.add_argument("--players",      default="players.json",
                   help="JSON file mapping tracker IDs to player names")
    p.add_argument("--max-frames",   type=int, default=0,
                   help="Stop after N frames (0 = all)")
    p.add_argument("--id-ref",       default="id_reference.jpg",
                   help="Path to save the tracker-ID reference image")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    player_names = load_player_names(args.players)
    if player_names:
        print(f"Loaded {len(player_names)} player name(s) from {args.players}")
    else:
        print(f"No players.json found — players will show as #ID numbers.")
        print(f"After this run, open {args.id_ref} to see IDs, then fill in players.json.\n")

    cap   = cv2.VideoCapture(args.video)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fw}x{fh} @ {fps:.1f}fps  ({total} frames)")

    writer    = make_writer(args.output, fps, fw, fh)
    detector  = LacrosseDetector()
    team_clf  = TeamClassifier()
    clf_ready = False
    id_ref_saved = False
    frame_idx = 0

    print(f"Processing → {args.output}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        detections = detector.detect(frame)
        players, _ = detector.split(detections)

        if len(players) == 0:
            writer.write(frame)
            frame_idx += 1
            continue

        bboxes    = players.xyxy
        track_ids = players.tracker_id
        if track_ids is None:
            track_ids = np.arange(len(bboxes))

        # Save ID reference on the first frame that has enough players
        if not id_ref_saved and len(bboxes) >= 2:
            save_id_reference(frame, bboxes, track_ids, args.id_ref)
            id_ref_saved = True

        if frame_idx % TEAM_REFIT_INTERVAL == 0:
            if team_clf.fit(frame, bboxes):
                clf_ready = True

        teams = team_clf.predict(frame, bboxes) if clf_ready \
                else np.zeros(len(bboxes), dtype=int)

        out_frame = draw_players(frame, bboxes, track_ids, teams,
                                 args.offense, player_names)
        out_frame = draw_legend(out_frame)
        writer.write(out_frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = (frame_idx / total * 100) if total else 0
            print(f"  Frame {frame_idx}/{total}  ({pct:.1f}%)")

    cap.release()
    writer.release()
    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
