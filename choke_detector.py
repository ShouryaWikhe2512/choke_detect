"""
Scene-aware traffic choke detection engine.

Pipeline:
1) Scene classification (arterial / intersection / merge / flyover)
2) Flow clustering (directional vehicle flows)
3) Spillback detection
4) Effective lane utilization
5) Choke classification (structural / spillback / behavioral / demand)
6) Clean visualization + structured JSON output
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO


# COCO vehicle and signal classes on YOLOv8 COCO model
# Include two-wheelers explicitly for mixed-traffic analysis.
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
TRAFFIC_SIGNAL_CLASS_IDS = {9, 11}  # traffic light, stop sign


def load_image(image_path: str) -> np.ndarray:
    """Load image from disk, exit with JSON error if missing."""
    path = Path(image_path)
    if not path.exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}), file=sys.stderr)
        sys.exit(1)
    img = cv2.imread(str(path))
    if img is None:
        print(json.dumps({"error": f"Could not read image: {image_path}"}), file=sys.stderr)
        sys.exit(1)
    return img


def run_yolo_vehicles(model: YOLO, img_bgr: np.ndarray, conf_threshold: float = 0.25):
    """
    Run YOLOv8 and return only vehicle and signal detections.
    Each detection: {"class_id", "bbox" (x1,y1,x2,y2), "center" (cx,cy), "confidence"}.
    """
    results = model(img_bgr, verbose=False)[0]
    boxes = results.boxes
    detections = []
    if boxes is None:
        return detections

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        if cls_id in TRAFFIC_SIGNAL_CLASS_IDS:
            # Allow lower threshold for signals; we will refine with NMS separately.
            if conf < min(0.15, conf_threshold):
                continue
        else:
            if conf < conf_threshold:
                continue
        if cls_id not in VEHICLE_CLASS_IDS and cls_id not in TRAFFIC_SIGNAL_CLASS_IDS:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        detections.append(
            {
                "class_id": cls_id,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "center": (float(cx), float(cy)),
                "confidence": conf,
            }
        )
    return detections


def split_vehicle_and_signal_detections(detections):
    vehicles = [d for d in detections if d["class_id"] in VEHICLE_CLASS_IDS]
    signals = [d for d in detections if d["class_id"] in TRAFFIC_SIGNAL_CLASS_IDS]
    return vehicles, signals


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5):
    """Simple class-agnostic NMS for traffic signal boxes."""
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def detect_traffic_signals_multi_scale(model: YOLO, img_bgr: np.ndarray):
    """
    Enhance traffic signal detection with simple multi-scale inference and NMS.
    Returns (signal_detected, signal_confidence).
    """
    scales = [1.0, 1.5, 2.0]
    all_boxes = []
    all_scores = []

    for s in scales:
        if s == 1.0:
            resized = img_bgr
        else:
            resized = cv2.resize(
                img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR
            )
        res = model(resized, verbose=False, conf=0.1, iou=0.5)[0]
        if res.boxes is None:
            continue
        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i].item())
            if cls_id not in TRAFFIC_SIGNAL_CLASS_IDS:
                continue
            conf = float(res.boxes.conf[i].item())
            if conf < 0.15:
                continue
            x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy()
            # Map back to original scale
            x1 /= s
            y1 /= s
            x2 /= s
            y2 /= s
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(conf)

    if not all_boxes:
        return False, 0.0

    boxes_np = np.array(all_boxes, dtype=np.float32)
    scores_np = np.array(all_scores, dtype=np.float32)
    keep = _nms_boxes(boxes_np, scores_np, iou_thresh=0.5)
    if not keep:
        return False, 0.0
    final_scores = scores_np[keep]
    signal_confidence = float(min(1.0, float(final_scores.max())))
    return True, signal_confidence


def detect_lane_lines(img_bgr: np.ndarray):
    """
    Detect dominant road line segments using Canny + probabilistic Hough.
    Returns list of segments and corresponding angles in degrees.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # Use full frame but emphasize lower half with a mild blur
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=min(h, w) // 6,
        maxLineGap=25,
    )

    if lines is None:
        return [], []

    segments = []
    angles_deg = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))  # from horizontal
        segments.append((float(x1), float(y1), float(x2), float(y2)))
        # Normalize angle into [-90, 90] for orientation
        ang_norm = ((angle + 90.0) % 180.0) - 90.0
        angles_deg.append(float(ang_norm))

    return segments, angles_deg


def estimate_vanishing_point(line_segments, img_shape):
    """
    Rough vanishing point estimation by clustering intersections of lane lines.
    Returns (vx, vy) in image coordinates or None if unreliable.
    """
    h, w = img_shape[:2]
    n = len(line_segments)
    if n < 3:
        return None

    intersections = []

    def segment_to_line(seg):
        x1, y1, x2, y2 = seg
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        return a, b, c

    lines = [segment_to_line(seg) for seg in line_segments]

    for i in range(n):
        a1, b1, c1 = lines[i]
        for j in range(i + 1, n):
            a2, b2, c2 = lines[j]
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-3:
                continue
            x = (b1 * c2 - b2 * c1) / det
            y = (c1 * a2 - c2 * a1) / det
            # Only consider intersections in front of camera (upper half-ish)
            if -0.25 * w <= x <= 1.25 * w and -0.25 * h <= y <= h * 1.0:
                intersections.append((x, y))

    if len(intersections) < 4:
        return None

    pts = np.array(intersections, dtype=np.float32)
    # Cluster intersections into one or two groups, keep main centroid
    k = 2 if len(intersections) >= 8 else 1
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pts)
    counts = [np.sum(labels == i) for i in range(k)]
    main_cluster = int(np.argmax(counts))
    vp = pts[labels == main_cluster].mean(axis=0)
    return float(vp[0]), float(vp[1])


def compute_birds_eye_view(img_bgr: np.ndarray):
    """
    Simple bird's-eye approximation using a fixed trapezoidal source region.
    Used for stop-line and zebra detection.
    """
    h, w = img_bgr.shape[:2]
    src = np.float32(
        [
            [w * 0.2, h * 0.7],
            [w * 0.8, h * 0.7],
            [w * 0.1, h * 0.98],
            [w * 0.9, h * 0.98],
        ]
    )
    dst_w = w
    dst_h = int(h * 0.6)
    dst = np.float32([[0, 0], [dst_w, 0], [0, dst_h], [dst_w, dst_h]])
    M = cv2.getPerspectiveTransform(src, dst)
    bird = cv2.warpPerspective(img_bgr, M, (dst_w, dst_h))
    return bird, M


def detect_stop_line(bird_bgr: np.ndarray) -> bool:
    """Detect strong horizontal stop line in bird's-eye view."""
    gray = cv2.cvtColor(bird_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    roi = gray[int(h * 0.1) : int(h * 0.6), :]
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=int(w * 0.3),
        maxLineGap=10,
    )
    if lines is None:
        return False
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = abs(np.degrees(np.arctan2(dy, dx)))
        if ang < 15.0:  # near horizontal
            return True
    return False


def detect_zebra_crossing(bird_bgr: np.ndarray) -> bool:
    """Detect repeated white stripes consistent with zebra crossing in bird's-eye view."""
    h, w = bird_bgr.shape[:2]
    hsv = cv2.cvtColor(bird_bgr, cv2.COLOR_BGR2HSV)
    # White-ish regions
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    band = mask[int(h * 0.1) : int(h * 0.6), :]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    stripe_count = 0
    area_min = (w * h) * 0.0002
    area_max = (w * h) * 0.05
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < area_min or area > area_max:
            continue
        if hh == 0:
            continue
        ratio = ww / float(hh)
        if 2.0 < ratio < 8.0:
            stripe_count += 1
    return stripe_count >= 3


def _point_segment_distance(px, py, x1, y1, x2, y2):
    """Euclidean distance from point to line segment."""
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-6:
        return np.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, (vx * wx + vy * wy) / seg_len2))
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return np.hypot(px - proj_x, py - proj_y)


def detect_traffic_poles(line_segments, img_shape):
    """
    Heuristic traffic-pole detection: tall near-vertical segments near the junction.
    Used only to support signalized_intersection classification.
    """
    h, w = img_shape[:2]
    if not line_segments:
        return False
    count = 0
    for x1, y1, x2, y2 in line_segments:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = abs(np.degrees(np.arctan2(dy, dx)))
        length = np.hypot(dx, dy)
        if ang > 75.0 and length > h * 0.15:
            yc = (y1 + y2) / 2.0
            if h * 0.2 <= yc <= h * 0.9:
                count += 1
    return count >= 2


def cluster_vehicle_flows(vehicles, line_segments, line_angles_deg):
    """
    Assign each vehicle a flow direction based on nearest lane line orientation.
    Cluster flows using KMeans on unit direction vectors.
    Returns:
      - flow_metrics: dict with dominant_flow_count, flow_alignment_score, cross_conflict_score
      - flows: list of {"direction_deg", "vehicle_count", "center": (cx, cy)}
      - per_vehicle_angles_rad: list of angles (radians) assigned to each vehicle (same order)
    """
    if not vehicles:
        return (
            {
                "dominant_flow_count": 0,
                "flow_alignment_score": 0.0,
                "cross_conflict_score": 0.0,
            },
            [],
            [],
        )

    # Precompute line angles in radians
    if line_segments and line_angles_deg:
        line_angles_rad = [np.deg2rad(a) for a in line_angles_deg]
    else:
        line_angles_rad = []

    def nearest_line_angle(cx, cy):
        if not line_segments or not line_angles_rad:
            return 0.0
        best_idx = 0
        best_dist = float("inf")
        for idx, seg in enumerate(line_segments):
            x1, y1, x2, y2 = seg
            d = _point_segment_distance(cx, cy, x1, y1, x2, y2)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return line_angles_rad[best_idx]

    angles_rad = []
    for v in vehicles:
        cx, cy = v["center"]
        ang = nearest_line_angle(cx, cy)
        angles_rad.append(float(ang))

    # Encode angles as unit vectors for clustering
    vecs = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
    n = vecs.shape[0]

    if n < 2:
        flow_alignment_score = 1.0
        cross_conflict_score = 0.0
        flows = [
            {
                "direction_deg": float(np.rad2deg(angles_rad[0])),
                "vehicle_count": n,
                "center": vehicles[0]["center"],
            }
        ]
        return (
            {
                "dominant_flow_count": 1,
                "flow_alignment_score": flow_alignment_score,
                "cross_conflict_score": cross_conflict_score,
            },
            flows,
            angles_rad,
        )

    n_clusters = min(3, n)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(vecs)

    flows = []
    cluster_angles = []
    dominant_flow_count = 0
    min_cluster_size = max(2, int(0.15 * n))

    for c in range(n_clusters):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        # Mean direction
        mean_vec = vecs[idxs].mean(axis=0)
        mean_ang = float(np.arctan2(mean_vec[1], mean_vec[0]))
        cluster_angles.append(mean_ang)
        # Mean center for arrow placement
        centers = np.array([vehicles[i]["center"] for i in idxs])
        center = (float(centers[:, 0].mean()), float(centers[:, 1].mean()))
        flows.append(
            {
                "direction_deg": float(np.rad2deg(mean_ang)),
                "vehicle_count": int(len(idxs)),
                "center": center,
            }
        )
        if len(idxs) >= min_cluster_size:
            dominant_flow_count += 1

    # Flow alignment: low angular variance => high alignment
    ang_array = np.array(angles_rad)
    # Normalize to [-pi/2, pi/2]
    ang_array = ((ang_array + np.pi / 2.0) % np.pi) - np.pi / 2.0
    if ang_array.size > 1:
        std_ang = float(np.std(ang_array))
        flow_alignment_score = max(0.0, 1.0 - std_ang / (np.pi / 2.0))
    else:
        flow_alignment_score = 1.0

    # Cross conflict: maximum pairwise separation between cluster means
    cross_conflict_score = 0.0
    if len(cluster_angles) > 1:
        max_delta = 0.0
        for i in range(len(cluster_angles)):
            for j in range(i + 1, len(cluster_angles)):
                d = abs(cluster_angles[i] - cluster_angles[j])
                d = min(d, np.pi - d)
                if d > max_delta:
                    max_delta = d
        cross_conflict_score = float(min(1.0, max_delta / (np.pi / 2.0)))

    flow_metrics = {
        "dominant_flow_count": int(dominant_flow_count),
        "flow_alignment_score": float(flow_alignment_score),
        "cross_conflict_score": float(cross_conflict_score),
    }
    return flow_metrics, flows, angles_rad


def load_video_frames(path: str, max_frames: int = 10):
    """Load up to max_frames from a video file. Returns list of BGR frames."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    if not cap.isOpened():
        return frames
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def compute_vehicle_speeds_from_flow(frames_bgr, vehicles):
    """
    Compute per-vehicle average motion magnitude using Farneback optical flow.
    Returns list of normalized speeds in [0, 1] aligned with vehicles list.
    """
    if not vehicles or len(frames_bgr) < 2:
        return [0.0 for _ in vehicles]

    gray_prev = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    h, w = gray_prev.shape[:2]

    speeds = np.zeros(len(vehicles), dtype=np.float32)
    steps = 0

    for frame in frames_bgr[1:]:
        gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev,
            gray_next,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        fh, fw = flow.shape[:2]
        for idx, v in enumerate(vehicles):
            cx, cy = v["center"]
            ix = int(np.clip(cx, 0, fw - 1))
            iy = int(np.clip(cy, 0, fh - 1))
            fx, fy = flow[iy, ix]
            speeds[idx] += np.hypot(fx, fy)
        steps += 1
        gray_prev = gray_next

    if steps > 0:
        speeds /= float(steps)
    # Normalize: assume ~5 px/frame as "free" motion upper bound
    norm_speeds = np.clip(speeds / 5.0, 0.0, 1.0)
    return norm_speeds.tolist()


def _rect_intersection_area(r1, r2):
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[2], r2[2])
    y2 = min(r1[3], r2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def compute_spillback_metrics(
    img_shape, vehicles, dominant_direction_deg: float | None = None, vehicle_speeds=None
):
    """
    Estimate exit clearance, downstream queue, and intersection blockage
    using simple rectangular regions in image space.
    """
    h, w = img_shape[:2]
    if h <= 0 or w <= 0:
        return {
            "exit_clearance_ratio": 1.0,
            "downstream_queue_length": 0.0,
            "intersection_blockage_ratio": 0.0,
            "spillback_flag": False,
            "conflict_stagnation_score": 0.0,
            "vehicles_in_conflict_zone": 0,
        }

    # Define regions: intersection (center box) and a dynamic exit region along dominant flow
    intersection_region = (
        int(w * 0.2),
        int(h * 0.3),
        int(w * 0.8),
        int(h * 0.6),
    )
    # Dynamic exit region
    if dominant_direction_deg is None:
        # Fallback: top-center band
        exit_region = (
            int(w * 0.25),
            0,
            int(w * 0.75),
            int(h * 0.25),
        )
    else:
        ang = dominant_direction_deg
        # Horizontal flow → exit on left or right edge
        if abs(ang) <= 45.0:
            # Determine left vs right bias by average vehicle x
            xs = [v["center"][0] for v in vehicles] if vehicles else [w * 0.5]
            mean_x = float(np.mean(xs))
            if mean_x >= w * 0.5:
                # Flow to the right
                exit_region = (int(w * 0.7), int(h * 0.2), w, int(h * 0.8))
            else:
                # Flow to the left
                exit_region = (0, int(h * 0.2), int(w * 0.3), int(h * 0.8))
        else:
            # Predominantly vertical flow → exit towards top
            exit_region = (
                int(w * 0.25),
                0,
                int(w * 0.75),
                int(h * 0.25),
            )

    exit_area = float((exit_region[2] - exit_region[0]) * (exit_region[3] - exit_region[1]))
    inter_area = float(
        (intersection_region[2] - intersection_region[0]) * (intersection_region[3] - intersection_region[1])
    )

    exit_occ = 0.0
    inter_occ = 0.0
    exit_centroids_y = []
    conflict_vehicle_count = 0
    conflict_indices = []

    for idx, v in enumerate(vehicles):
        x1, y1, x2, y2 = v["bbox"]
        bbox = (x1, y1, x2, y2)
        exit_occ += _rect_intersection_area(bbox, exit_region)
        inter_occ += _rect_intersection_area(bbox, intersection_region)
        cx, cy = v["center"]
        if exit_region[0] <= cx <= exit_region[2] and exit_region[1] <= cy <= exit_region[3]:
            exit_centroids_y.append(cy)
        if (
            intersection_region[0] <= cx <= intersection_region[2]
            and intersection_region[1] <= cy <= intersection_region[3]
        ):
            conflict_vehicle_count += 1
            conflict_indices.append(idx)

    exit_occ_ratio = float(min(1.0, exit_occ / exit_area)) if exit_area > 0 else 0.0
    intersection_blockage_ratio = float(min(1.0, inter_occ / inter_area)) if inter_area > 0 else 0.0
    exit_clearance_ratio = float(max(0.0, 1.0 - exit_occ_ratio))

    downstream_queue_length = 0.0
    if exit_centroids_y:
        # Vertical queue extent in pixels inside exit region
        y_min = min(exit_centroids_y)
        y_max = max(exit_centroids_y)
        downstream_queue_length = float(max(0.0, y_max - y_min))

    # Conflict stagnation: prefer real motion from optical flow when available.
    avg_motion = None
    if vehicle_speeds is not None and conflict_indices:
        speeds_conf = [vehicle_speeds[i] for i in conflict_indices]
        if speeds_conf:
            avg_motion = float(np.mean(speeds_conf))
    if avg_motion is None:
        # Fallback proxy from available free space
        avg_motion = max(0.0, 1.0 - intersection_blockage_ratio)
    vehicle_count = len(vehicles)
    min_conflict_vehicles = max(3, int(0.15 * vehicle_count)) if vehicle_count > 0 else 0
    stagnation_vehicles = conflict_vehicle_count >= min_conflict_vehicles and conflict_vehicle_count > 0
    stagnation_low_motion = avg_motion < 0.2
    # Score in [0, 1]: higher when more vehicles and lower free space.
    conflict_stagnation_score = 0.0
    if conflict_vehicle_count > 0:
        density_factor = min(1.0, conflict_vehicle_count / 10.0)
        conflict_stagnation_score = float(
            density_factor * max(0.0, 1.0 - avg_motion)
        )

    spillback_geom = exit_clearance_ratio < 0.3 and intersection_blockage_ratio > 0.3
    spillback_stagnation = stagnation_vehicles and stagnation_low_motion
    spillback_flag = bool(spillback_geom or spillback_stagnation)

    return {
        "exit_clearance_ratio": round(exit_clearance_ratio, 3),
        "downstream_queue_length": round(downstream_queue_length, 1),
        "intersection_blockage_ratio": round(intersection_blockage_ratio, 3),
        "spillback_flag": spillback_flag,
        "conflict_stagnation_score": float(round(conflict_stagnation_score, 3)),
        "vehicles_in_conflict_zone": int(conflict_vehicle_count),
    }


def compute_vehicle_density(vehicles, img_shape, road_region_fraction: float = 0.5) -> float:
    """Approximate normalized vehicle density 0–1 over lower road region."""
    h, w = img_shape[:2]
    if h <= 0 or w <= 0:
        return 0.0
    road_area = h * w * road_region_fraction
    if road_area <= 0:
        return 0.0
    vehicle_count = len(vehicles)
    vehicles_per_1k = (vehicle_count / road_area) * 1000.0
    density = min(1.0, vehicles_per_1k / 0.7)  # 0.7 vehicles / 1k px ~ high
    return float(round(density, 4))


def estimate_lane_utilization(
    vehicles, img_shape, flow_alignment_score: float, vanishing_point=None
):
    """
    Estimate utilized_lane_count and effective_lane_utilization in a road-aligned frame.
    """
    h, w = img_shape[:2]
    if not vehicles or h <= 0 or w <= 0:
        return {
            "utilized_lane_count": 0,
            "lane_alignment_score": float(flow_alignment_score),
            "effective_lane_utilization": 0.0,
        }

    # Rotate coordinates so that the road direction is approximately vertical.
    cx0, cy0 = w * 0.5, float(h)  # origin at bottom center
    if vanishing_point is not None:
        vx, vy = vanishing_point
        angle_dir = np.arctan2(vy - cy0, vx - cx0)
    else:
        # Fallback: assume road roughly vertical
        angle_dir = -np.pi / 2.0
    theta = -(angle_dir - (-np.pi / 2.0))  # rotation to make road vertical

    xs_rot = []
    for v in vehicles:
        cx, cy = v["center"]
        # Only consider vehicles in lower half as approach
        if cy < h * 0.5:
            continue
        dx = cx - cx0
        dy = cy - cy0
        xr = dx * np.cos(theta) - dy * np.sin(theta)
        xs_rot.append(xr)

    if not xs_rot:
        xs_rot = [v["center"][0] - cx0 for v in vehicles]

    xs_arr = np.array(xs_rot).reshape(-1, 1)
    n = xs_arr.shape[0]
    if n < 2:
        utilized_lane_count = 1
    else:
        n_clusters = min(4, n)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(xs_arr)
        sizes = [np.sum(labels == k) for k in range(n_clusters)]
        min_lane_size = max(2, int(0.1 * n))
        utilized_lane_count = sum(size >= min_lane_size for size in sizes)
        if utilized_lane_count == 0:
            utilized_lane_count = 1

    # Effective number of lanes allowed by width in rotated frame
    width_est = max(xs_rot) - min(xs_rot) if xs_rot else w
    potential_lanes = max(1, min(6, int(abs(width_est) / 200.0))) if width_est != 0 else 1
    effective_lane_utilization = float(
        min(1.0, utilized_lane_count / float(potential_lanes))
    )

    return {
        "utilized_lane_count": int(utilized_lane_count),
        "lane_alignment_score": float(flow_alignment_score),
        "effective_lane_utilization": round(effective_lane_utilization, 3),
    }


def classify_scene_type(
    line_angles_deg,
    dominant_flow_count: int,
    cross_conflict_score: float,
    traffic_signal_present: bool,
    intersection_blockage_ratio: float,
    img_shape,
    line_segments,
):
    """
    Classify scene into:
      - arterial_corridor
      - signalized_intersection
      - unsignalized_intersection
      - merge
      - flyover_approach
    """
    h, _ = img_shape[:2]
    angles = np.array(line_angles_deg, dtype=float)
    multi_orientation = False

    if angles.size >= 8:
        # KMeans into 2 orientation clusters and check separation
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        labels = kmeans.fit_predict(angles.reshape(-1, 1))
        centers = kmeans.cluster_centers_.flatten()
        delta = abs(centers[0] - centers[1])
        delta = min(delta, 180.0 - delta)
        multi_orientation = delta > 40.0
    else:
        multi_orientation = dominant_flow_count >= 2 and cross_conflict_score > 0.5

    # Approximate elevated structure detection for flyover
    upper_line_count = 0
    lower_line_count = 0
    for (x1, y1, x2, y2) in line_segments:
        yc = (y1 + y2) / 2.0
        if yc < h * 0.35:
            upper_line_count += 1
        elif yc > h * 0.55:
            lower_line_count += 1

    # Intersection classification
    if multi_orientation and intersection_blockage_ratio >= 0.15:
        if traffic_signal_present:
            return "signalized_intersection"
        return "unsignalized_intersection"

    # Merge: two flows with moderate angular separation but not full cross
    if dominant_flow_count >= 2 and 0.2 < cross_conflict_score < 0.7:
        return "merge"

    # Flyover approach: strong structure above, fewer lines below, low central blockage
    if upper_line_count > lower_line_count and intersection_blockage_ratio < 0.2:
        return "flyover_approach"

    # Default arterial corridor
    return "arterial_corridor"


def compute_lane_discipline_score(vehicles, per_vehicle_angles_rad):
    """
    Mixed-traffic lane discipline heuristic in [0, 1].
    Lower when many vehicles, especially two-wheelers, deviate diagonally from main flow.
    """
    if not vehicles or not per_vehicle_angles_rad:
        return 1.0

    angs = np.array(per_vehicle_angles_rad, dtype=np.float32)
    # Main flow direction
    main_dir = float(np.median(angs))

    def wrapped_delta(a):
        d = abs(a - main_dir)
        d = min(d, np.pi - d)
        return d

    deltas = np.array([wrapped_delta(a) for a in angs], dtype=np.float32)
    # Normalize by 45 degrees
    overall_alignment = float(max(0.0, 1.0 - (deltas.mean() / (np.pi / 4.0))))

    # Two-wheelers (bicycle + motorcycle)
    two_idx = [
        i
        for i, v in enumerate(vehicles)
        if v["class_id"] in {1, 3}
    ]
    if two_idx:
        deltas_two = deltas[two_idx]
        two_alignment = float(max(0.0, 1.0 - (deltas_two.mean() / (np.pi / 4.0))))
    else:
        two_alignment = overall_alignment

    lane_discipline = 0.5 * overall_alignment + 0.5 * two_alignment
    return float(max(0.0, min(1.0, lane_discipline)))


def classify_choke(
    scene_type: str,
    spillback_flag: bool,
    flow_alignment_score: float,
    cross_conflict_score: float,
    vehicle_density: float,
    intersection_blockage_ratio: float,
    effective_lane_utilization: float,
    conflict_stagnation_score: float,
    signal_present: bool,
):
    """
    Priority-based choke classification with sigmoid severity model.
    Types:
      - signal_noncompliance_gridlock
      - behavioral_gridlock
      - spillback_congestion
      - structural_geometry
      - demand_surge
    """
    reasons = []

    structural_geometry_flag = effective_lane_utilization < 0.5 and vehicle_density > 0.5

    # Priority hierarchy from specification
    if intersection_blockage_ratio > 0.7 and signal_present:
        choke_type = "signal_noncompliance_gridlock"
    elif intersection_blockage_ratio > 0.7:
        choke_type = "behavioral_gridlock"
    elif spillback_flag:
        choke_type = "spillback_congestion"
    elif structural_geometry_flag:
        choke_type = "structural_geometry"
    else:
        choke_type = "demand_surge"

    # Sigmoid-based choke score
    z = (
        2.0 * intersection_blockage_ratio
        + 1.5 * max(0.0, min(1.0, conflict_stagnation_score))
        + 1.0 * vehicle_density
        + 1.0 * (1.0 - flow_alignment_score)
    )
    score = float(1.0 / (1.0 + float(np.exp(-z))))

    # Diagnosis text per choke type
    if choke_type == "signal_noncompliance_gridlock":
        reasons.append(
            "Severe blocking of the signalized conflict zone with vehicles occupying the box despite limited exit clearance."
        )
        reasons.append(
            "Likely signal non-compliance or insufficient red-clear time causing standing queues across the stop line."
        )
    elif choke_type == "behavioral_gridlock":
        reasons.append(
            "Conflict zone is heavily blocked with simultaneous cross-direction occupancy and poor exit clearance."
        )
    elif choke_type == "spillback_congestion":
        reasons.append(
            "Downstream queues appear to spill back into the junction, restricting exit lanes and discharge."
        )
    elif choke_type == "structural_geometry":
        reasons.append(
            "Geometry and lane utilization limit throughput under current demand, reducing effective lane use."
        )
    else:  # demand_surge
        reasons.append(
            "High demand relative to available capacity with generally aligned flows and limited structural conflicts."
        )
    diagnosis = " ".join(reasons)

    if choke_type == "signal_noncompliance_gridlock":
        suggested_fix = (
            "Review and enforce signal compliance, introduce box markings and all-red or clearance intervals, "
            "and coordinate upstream signals to prevent vehicles entering the junction without downstream space."
        )
    elif choke_type == "spillback_congestion":
        suggested_fix = (
            "Increase downstream discharge capacity or storage, protect exit lanes, and "
            "apply signal or metering control upstream to prevent queues blocking the junction."
        )
    elif choke_type == "behavioral_gridlock":
        suggested_fix = (
            "Introduce clear conflict-zone markings and 'do not block' control, enforce yielding rules, "
            "and adjust lane use/signing to separate crossing and turning movements while preserving exit clearance."
        )
    elif choke_type == "structural_geometry":
        suggested_fix = (
            "Reconfigure lane geometry and merges to better align flows, add or extend turn/through lanes, "
            "and remove bottleneck pinch points."
        )
    else:  # demand_surge
        suggested_fix = (
            "Provide additional capacity or apply demand management (e.g., peak-period signal plans, "
            "ramp metering, or access control) to handle peak surges."
        )

    return choke_type, score, diagnosis, suggested_fix


def estimate_congestion_region(img_shape, vehicles):
    """Bounding rectangle around vehicles with padding for congestion overlay."""
    h, w = img_shape[:2]
    if not vehicles:
        return (0, 0, w, h)
    xs = [v["center"][0] for v in vehicles]
    ys = [v["center"][1] for v in vehicles]
    x1 = max(0, int(min(xs) - w * 0.05))
    x2 = min(w, int(max(xs) + w * 0.05))
    y1 = max(0, int(min(ys) - h * 0.05))
    y2 = min(h, int(max(ys) + h * 0.05))
    return (x1, y1, x2, y2)


def render_overlay(
    img_bgr: np.ndarray,
    congestion_region,
    scene_type: str,
    choke_type: str,
    choke_score: float,
    intersection_blockage_ratio: float,
    exit_clearance_ratio: float,
    signal_detected: bool,
):
    """
    Draw a clean, minimal overlay:
      - Semi-transparent red region over congestion/conflict area
      - Optional highlight of blocked exit region
      - Scene + choke labels and score in a top-left text panel
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Congestion region
    x1, y1, x2, y2 = congestion_region
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), thickness=-1)
    alpha = 0.25
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    # Optional blocked-exit highlight (top-center band)
    exit_region = (
        int(w * 0.25),
        0,
        int(w * 0.75),
        int(h * 0.25),
    )
    if exit_clearance_ratio < 0.3 or intersection_blockage_ratio > 0.7:
        overlay_exit = out.copy()
        cv2.rectangle(
            overlay_exit,
            (exit_region[0], exit_region[1]),
            (exit_region[2], exit_region[3]),
            (0, 165, 255),
            thickness=-1,
        )
        out = cv2.addWeighted(overlay_exit, 0.35, out, 0.65, 0)

    # Labels (minimal, production-style)
    label_bg_color = (0, 0, 0)
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    label1 = f"Scene: {scene_type}"
    label2 = f"Choke: {choke_type}"
    label3 = f"Score: {choke_score:.2f}"
    label4 = f"Signal: {'Yes' if signal_detected else 'No'}"

    def draw_label(text, y):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(out, (10, y - th - 6), (10 + tw + 6, y + 4), label_bg_color, -1)
        cv2.putText(out, text, (13, y), font, scale, text_color, thickness, cv2.LINE_AA)

    draw_label(label1, 25)
    draw_label(label2, 50)
    draw_label(label3, 75)
    draw_label(label4, 100)

    # Small signal indicator icon in top-right
    icon_center = (w - 30, 30)
    radius = 10
    color_on = (0, 255, 0)
    color_off = (128, 128, 128)
    cv2.circle(out, icon_center, radius, color_on if signal_detected else color_off, -1)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Scene-aware traffic choke detection from a single road image."
    )
    parser.add_argument("image_path", type=str, help="Path to input road image or video")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model path (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip interactive window display (still saves annotated image).",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Interpret image_path as video and use multiple frames for optical flow.",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=8,
        help="Maximum number of frames to read from video (default: 8).",
    )
    args = parser.parse_args()

    # Media loading (single frame or short clip)
    if args.video:
        frames_bgr = load_video_frames(args.image_path, max_frames=args.max_video_frames)
        if not frames_bgr:
            print(
                json.dumps({"error": f"Could not read video: {args.image_path}"}),
                file=sys.stderr,
            )
            sys.exit(1)
        img = frames_bgr[0]
    else:
        img = load_image(args.image_path)
        frames_bgr = [img]

    model = YOLO(args.model)

    # STEP 1 & 2: detections, lane lines, and flow clustering
    all_dets = run_yolo_vehicles(model, img)
    vehicles, signals = split_vehicle_and_signal_detections(all_dets)
    line_segments, line_angles_deg = detect_lane_lines(img)

    flow_metrics, flows, per_vehicle_angles_rad = cluster_vehicle_flows(
        vehicles, line_segments, line_angles_deg
    )

    # Enhanced traffic signal detection (multi-scale)
    traffic_signal_detected, signal_confidence = detect_traffic_signals_multi_scale(
        model, img
    )

    # Optional traffic pole heuristic to support signalized scene classification
    traffic_pole_present = detect_traffic_poles(line_segments, img.shape)

    # Vanishing point for perspective-aware geometry
    vanishing_point = estimate_vanishing_point(line_segments, img.shape)

    # Optional temporal intelligence via optical flow
    vehicle_speeds = compute_vehicle_speeds_from_flow(frames_bgr, vehicles)
    dominant_direction_deg = None
    if flows:
        dominant_direction_deg = sorted(
            flows, key=lambda x: x["vehicle_count"], reverse=True
        )[0]["direction_deg"]

    # STEP 3: Spillback metrics (dynamic exit + real motion if available)
    spillback_metrics = compute_spillback_metrics(
        img.shape,
        vehicles,
        dominant_direction_deg=dominant_direction_deg,
        vehicle_speeds=vehicle_speeds,
    )

    # STEP 4: Lane utilization (using perspective-aware clustering)
    vehicle_density = compute_vehicle_density(vehicles, img.shape)
    lane_metrics = estimate_lane_utilization(
        vehicles,
        img.shape,
        flow_metrics["flow_alignment_score"],
        vanishing_point=vanishing_point,
    )
    lane_discipline_score = compute_lane_discipline_score(
        vehicles, per_vehicle_angles_rad
    )

    # Infrastructure-aware elements (stop line, zebra crossing)
    bird_view, _ = compute_birds_eye_view(img)
    stop_line_detected = detect_stop_line(bird_view)
    zebra_crossing_detected = detect_zebra_crossing(bird_view)

    # STEP 1 (scene) – using flows + blockage + confirmed signals/poles
    traffic_signal_present = bool(traffic_signal_detected or traffic_pole_present)
    scene_type = classify_scene_type(
        line_angles_deg=line_angles_deg,
        dominant_flow_count=flow_metrics["dominant_flow_count"],
        cross_conflict_score=flow_metrics["cross_conflict_score"],
        traffic_signal_present=traffic_signal_present,
        intersection_blockage_ratio=spillback_metrics["intersection_blockage_ratio"],
        img_shape=img.shape,
        line_segments=line_segments,
    )

    # STEP 5: Choke classification (no merge-angle logic; scene-aware)
    choke_type, choke_score, diagnosis, suggested_fix = classify_choke(
        scene_type=scene_type,
        spillback_flag=spillback_metrics["spillback_flag"],
        flow_alignment_score=flow_metrics["flow_alignment_score"],
        cross_conflict_score=flow_metrics["cross_conflict_score"],
        vehicle_density=vehicle_density,
        intersection_blockage_ratio=spillback_metrics["intersection_blockage_ratio"],
        effective_lane_utilization=lane_metrics["effective_lane_utilization"],
        conflict_stagnation_score=spillback_metrics["conflict_stagnation_score"],
        signal_present=traffic_signal_present,
    )

    # STEP 6: Visualization – clean overlay only
    congestion_region = estimate_congestion_region(img.shape, vehicles)
    annotated = render_overlay(
        img_bgr=img,
        congestion_region=congestion_region,
        scene_type=scene_type,
        choke_type=choke_type,
        choke_score=choke_score,
        intersection_blockage_ratio=spillback_metrics["intersection_blockage_ratio"],
        exit_clearance_ratio=spillback_metrics["exit_clearance_ratio"],
        signal_detected=traffic_signal_detected,
    )

    # STEP 7: Structured JSON output
    dominant_flows_summary = [
        {
            "direction_deg": float(f["direction_deg"]),
            "vehicle_count": int(f["vehicle_count"]),
        }
        for f in sorted(flows, key=lambda x: x["vehicle_count"], reverse=True)
    ]

    output = {
        "scene_type": scene_type,
        "choke_type": choke_type,
        "choke_score": float(round(choke_score, 3)),
        "dominant_flows": dominant_flows_summary,
        "spillback_flag": bool(spillback_metrics["spillback_flag"]),
        "intersection_blockage_ratio": float(
            round(spillback_metrics["intersection_blockage_ratio"], 3)
        ),
        "flow_alignment_score": float(round(flow_metrics["flow_alignment_score"], 3)),
        "vehicle_density": float(vehicle_density),
        "lane_discipline_score": float(round(lane_discipline_score, 3)),
        "traffic_signal_detected": bool(traffic_signal_detected),
        "stop_line_detected": bool(stop_line_detected),
        "zebra_crossing_detected": bool(zebra_crossing_detected),
        "signal_confidence": float(round(signal_confidence, 3)),
        "diagnosis": diagnosis,
        "suggested_fix": suggested_fix,
    }

    print(json.dumps(output, indent=2))

    # Save annotated image
    in_path = Path(args.image_path)
    out_path = in_path.parent / f"{in_path.stem}_scene_choke{in_path.suffix}"
    cv2.imwrite(str(out_path), annotated)
    print(f"Annotated scene written to: {out_path}", file=sys.stderr)

    if not args.no_display:
        cv2.imshow("Scene-aware choke diagnosis", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

