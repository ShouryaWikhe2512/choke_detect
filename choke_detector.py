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
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
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


def _rect_intersection_area(r1, r2):
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[2], r2[2])
    y2 = min(r1[3], r2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def compute_spillback_metrics(img_shape, vehicles):
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

    # Define regions: exit (top center), intersection (center box)
    exit_region = (
        int(w * 0.25),
        0,
        int(w * 0.75),
        int(h * 0.25),
    )
    intersection_region = (
        int(w * 0.2),
        int(h * 0.3),
        int(w * 0.8),
        int(h * 0.6),
    )

    exit_area = float((exit_region[2] - exit_region[0]) * (exit_region[3] - exit_region[1]))
    inter_area = float(
        (intersection_region[2] - intersection_region[0]) * (intersection_region[3] - intersection_region[1])
    )

    exit_occ = 0.0
    inter_occ = 0.0
    exit_centroids_y = []
    conflict_vehicle_count = 0

    for v in vehicles:
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

    exit_occ_ratio = float(min(1.0, exit_occ / exit_area)) if exit_area > 0 else 0.0
    intersection_blockage_ratio = float(min(1.0, inter_occ / inter_area)) if inter_area > 0 else 0.0
    exit_clearance_ratio = float(max(0.0, 1.0 - exit_occ_ratio))

    downstream_queue_length = 0.0
    if exit_centroids_y:
        # Vertical queue extent in pixels inside exit region
        y_min = min(exit_centroids_y)
        y_max = max(exit_centroids_y)
        downstream_queue_length = float(max(0.0, y_max - y_min))

    # Conflict stagnation proxy: many vehicles in conflict zone with very low "motion".
    # Since we only have a single frame, we approximate average motion magnitude
    # as the free space in the conflict zone: 1 - intersection_blockage_ratio.
    avg_motion = max(0.0, 1.0 - intersection_blockage_ratio)
    vehicle_count = len(vehicles)
    min_conflict_vehicles = max(3, int(0.15 * vehicle_count)) if vehicle_count > 0 else 0
    stagnation_vehicles = conflict_vehicle_count >= min_conflict_vehicles and conflict_vehicle_count > 0
    stagnation_low_motion = avg_motion < 0.2
    # Score in [0, 1]: higher when more vehicles and lower free space.
    conflict_stagnation_score = 0.0
    if conflict_vehicle_count > 0:
        density_factor = min(1.0, conflict_vehicle_count / 10.0)
        conflict_stagnation_score = float(density_factor * (1.0 - avg_motion))

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


def estimate_lane_utilization(vehicles, img_shape, flow_alignment_score: float):
    """
    Estimate utilized_lane_count and effective_lane_utilization from vehicle rows.
    """
    h, w = img_shape[:2]
    if not vehicles or h <= 0 or w <= 0:
        return {
            "utilized_lane_count": 0,
            "lane_alignment_score": float(flow_alignment_score),
            "effective_lane_utilization": 0.0,
        }

    # Use vehicles in lower half (approach region)
    xs = []
    for v in vehicles:
        cx, cy = v["center"]
        if cy >= h * 0.5:
            xs.append(cx)
    if not xs:
        xs = [v["center"][0] for v in vehicles]

    xs_arr = np.array(xs).reshape(-1, 1)
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

    potential_lanes = max(1, min(6, int(w / 200)))
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


def classify_choke(
    scene_type: str,
    spillback_flag: bool,
    flow_alignment_score: float,
    cross_conflict_score: float,
    vehicle_density: float,
    intersection_blockage_ratio: float,
    effective_lane_utilization: float,
    conflict_stagnation_score: float,
):
    """
    Priority-based choke classification with severity aligned to blockage and conflicts.
    Types:
      - structural_geometry
      - spillback_congestion
      - behavioral_gridlock
      - demand_surge
    """
    reasons = []

    structural_geometry_flag = effective_lane_utilization < 0.5 and vehicle_density > 0.5

    # Priority hierarchy:
    # 1) Severe conflict-zone blockage
    if intersection_blockage_ratio > 0.7:
        choke_type = "behavioral_gridlock"
    # 2) Downstream spillback into the node
    elif spillback_flag:
        choke_type = "spillback_congestion"
    # 3) Constrained geometry under demand
    elif structural_geometry_flag:
        choke_type = "structural_geometry"
    # 4) Residual high loading
    else:
        choke_type = "demand_surge"

    # Unified choke severity score
    score = (
        0.4 * intersection_blockage_ratio
        + 0.2 * vehicle_density
        + 0.2 * (1.0 - flow_alignment_score)
        + 0.2 * max(0.0, min(1.0, conflict_stagnation_score))
    )
    score = float(max(0.0, min(1.0, score)))
    if intersection_blockage_ratio > 0.8 and score < 0.75:
        score = 0.75

    # Diagnosis text per choke type
    if choke_type == "behavioral_gridlock":
        reasons.append(
            "Conflict zone is heavily blocked with simultaneous cross-direction occupancy and poor exit clearance."
        )
        if intersection_blockage_ratio > 0.9:
            reasons.append("Intersection is almost fully occupied (blockage ratio close to 1.0).")
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

    if choke_type == "spillback_congestion":
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

    def draw_label(text, y):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(out, (10, y - th - 6), (10 + tw + 6, y + 4), label_bg_color, -1)
        cv2.putText(out, text, (13, y), font, scale, text_color, thickness, cv2.LINE_AA)

    draw_label(label1, 25)
    draw_label(label2, 50)
    draw_label(label3, 75)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Scene-aware traffic choke detection from a single road image."
    )
    parser.add_argument("image_path", type=str, help="Path to input road image")
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
    args = parser.parse_args()

    img = load_image(args.image_path)
    model = YOLO(args.model)

    # STEP 1 & 2: detections, lane lines, and flow clustering
    all_dets = run_yolo_vehicles(model, img)
    vehicles, signals = split_vehicle_and_signal_detections(all_dets)
    line_segments, line_angles_deg = detect_lane_lines(img)

    flow_metrics, flows, per_vehicle_angles_rad = cluster_vehicle_flows(
        vehicles, line_segments, line_angles_deg
    )

    # STEP 3: Spillback metrics
    spillback_metrics = compute_spillback_metrics(img.shape, vehicles)

    # STEP 4: Lane utilization (using flow alignment as lane_alignment_score)
    vehicle_density = compute_vehicle_density(vehicles, img.shape)
    lane_metrics = estimate_lane_utilization(
        vehicles, img.shape, flow_metrics["flow_alignment_score"]
    )

    # STEP 1 (scene) – using flows + blockage + signals
    traffic_signal_present = bool(signals)
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

