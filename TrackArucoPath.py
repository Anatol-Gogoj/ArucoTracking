#!/usr/bin/env python3
"""
TrackArucoPath.py
=================
Detects and tracks a single ArUco fiducial (default ID 4) from a top-down video, draws the
center path as a blue line with a black outline, and overlays a projected 3D axis on the marker.
Writes an annotated MP4 and a CSV log of poses.

Installation
------------
    pip install opencv-contrib-python numpy

Quick Start
-----------
With real camera calibration (recommended):
    python TrackArucoPath.py --input input.mp4 --output output_annotated.mp4 \
        --camera CameraParams.npz --marker-length 0.05 --csv output_poses.csv

Without calibration (falls back to ~60° FOV guess):
    python TrackArucoPath.py --input input.mp4 --csv poses.csv

Minimal (auto output paths):
    python TrackArucoPath.py --input input.mp4 --marker-id 4

Arguments
---------
--input            Path to input video (required).
--output           Path to output MP4. Default: <input_basename>_annotated.mp4
--csv              Path to output CSV. Default: <input_basename>_poses.csv
--marker-id        Integer ArUco ID to track. Default: 4
--marker-length    Marker physical side length in meters for pose. Default: 0.05
--axis-length      Length of drawn axis in meters. Default: 0.5 * marker-length
--camera           Path to .npz with 'camera_matrix' (3x3) and 'dist_coeffs' (Nx1).
--dict             ArUco dictionary name. Default: DICT_4X4_50
--trail-max        Max number of stored points for the path. Default: 100000
--fov-deg          If no --camera is given, guess intrinsics using this FOV. Default: 60.0

CSV Columns
-----------
frame_index,time_s,found,marker_id,cx,cy,x,y,z,A_deg,B_deg,C_deg,rvec_x,rvec_y,rvec_z

Notes
-----
• A,B,C are rotations about X,Y,Z respectively (roll, pitch, yaw) in degrees, extracted from the
  rotation matrix with ZYX (yaw–pitch–roll) convention:
    A = atan2(R32, R33),  B = -asin(R31),  C = atan2(R21, R11)
• If your OpenCV build lacks cv2.aruco.drawAxis, this script uses cv2.drawFrameAxes, and if that
  is also unavailable, it manually projects axes with cv2.projectPoints.
• For accurate axis overlay and metric translations, provide a real calibration via --camera.
"""

import argparse
import csv
import math
import os
from collections import deque

import cv2
import numpy as np


def BuildDictionary(dict_name: str):
    name_map = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    if dict_name not in name_map:
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
    return cv2.aruco.getPredefinedDictionary(name_map[dict_name])


def LoadCameraParams(path: str):
    data = np.load(path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    return camera_matrix, dist_coeffs


def GuessCameraParams(frame_width: int, frame_height: int, fov_deg: float = 60.0):
    # Create plausible intrinsics if no calibration file is given.
    fov_rad = np.deg2rad(fov_deg)
    fx = 0.5 * frame_width / np.tan(0.5 * fov_rad)
    fy = fx
    cx = frame_width / 2.0
    cy = frame_height / 2.0
    camera_matrix = np.array([[fx, 0,  cx],
                              [0,  fy, cy],
                              [0,  0,   1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


def DetectMarkers(gray, dictionary, parameters):
    # OpenCV API changed around 4.7.0; support both.
    try:
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids, rejected


def _NormalizeRt(rvec, tvec):
    rvec = np.asarray(rvec).reshape(3, 1).astype(np.float32)
    tvec = np.asarray(tvec).reshape(3, 1).astype(np.float32)
    return rvec, tvec


def DrawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_len):
    """
    Draw axes using the most compatible API available:
    1) cv2.drawFrameAxes (calib3d)
    2) cv2.aruco.drawAxis (older contrib)
    3) Manual projection with cv2.projectPoints
    """
    rvec, tvec = _NormalizeRt(rvec, tvec)

    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, float(axis_len))
        return

    if hasattr(cv2.aruco, "drawAxis"):
        cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, float(axis_len))
        return

    # Manual fallback: project 3D axis endpoints and draw lines (X=red, Y=green, Z=blue).
    axis = np.float32([[0, 0, 0],
                       [axis_len, 0, 0],
                       [0, axis_len, 0],
                       [0, 0, -axis_len]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    origin = tuple(imgpts[0])
    cv2.line(frame, origin, tuple(imgpts[1]), (0, 0, 255), 3, cv2.LINE_AA)   # X (red)
    cv2.line(frame, origin, tuple(imgpts[2]), (0, 255, 0), 3, cv2.LINE_AA)   # Y (green)
    cv2.line(frame, origin, tuple(imgpts[3]), (255, 0, 0), 3, cv2.LINE_AA)   # Z (blue)


def ComputeCenter(corners):
    # corners shape: (1,4,2) or (4,2). Average the 4 image points.
    pts = np.asarray(corners).reshape(-1, 2)
    c = pts.mean(axis=0)
    return int(round(c[0])), int(round(c[1]))


def RotationMatrixToABC(R: np.ndarray):
    """
    Convert rotation matrix to A,B,C in degrees, where:
      A = rotation about X (roll)
      B = rotation about Y (pitch)
      C = rotation about Z (yaw)
    Using ZYX (yaw-pitch-roll) Tait–Bryan convention:
      A = atan2(R[2,1], R[2,2])
      B = -asin(R[2,0])
      C = atan2(R[1,0], R[0,0])
    Handles numerical edge cases with clamping.
    """
    R = np.asarray(R, dtype=np.float64)
    # Clamp to handle numeric issues
    r20 = np.clip(R[2, 0], -1.0, 1.0)
    B = -math.asin(r20)

    # Check for gimbal lock (cos(B) ~ 0)
    if abs(math.cos(B)) < 1e-8:
        # Gimbal lock: set A = 0, compute C from other terms
        A = 0.0
        C = math.atan2(-R[0, 1], R[1, 1])
    else:
        A = math.atan2(R[2, 1], R[2, 2])
        C = math.atan2(R[1, 0], R[0, 0])

    # Convert to degrees
    A = math.degrees(A)
    B = math.degrees(B)
    C = math.degrees(C)
    return A, B, C


def Main():
    parser = argparse.ArgumentParser(
        description="Track an ArUco marker and draw its center path + 3D axis overlay, with CSV logging.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input video path (mp4/mkv/etc).")
    parser.add_argument("--output", default=None, help="Output MP4 path. Default: <input>_annotated.mp4")
    parser.add_argument("--csv", default=None, help="Output CSV path. Default: <input>_poses.csv")
    parser.add_argument("--marker-id", type=int, default=4, help="Aruco marker ID to track. Default 4.")
    parser.add_argument("--marker-length", type=float, default=0.05, help="Marker length in meters. Default 0.05 m.")
    parser.add_argument("--axis-length", type=float, default=None, help="Axis length in meters. Default 0.5 * marker-length.")
    parser.add_argument("--camera", default=None, help="Path to .npz with camera_matrix and dist_coeffs.")
    parser.add_argument("--dict", dest="dict_name", default="DICT_4X4_50", help="Aruco dictionary name. Default DICT_4X4_50.")
    parser.add_argument("--trail-max", type=int, default=100000, help="Max number of points stored for the path.")
    parser.add_argument("--fov-deg", type=float, default=60.0, help="Fallback FOV if no camera file. Default 60 deg.")
    args = parser.parse_args()

    input_path = args.input
    base, _ = os.path.splitext(input_path)
    output_path = args.output or f"{base}_annotated.mp4"
    csv_path = args.csv or f"{base}_poses.csv"
    marker_id = args.marker_id
    marker_len = float(args.marker_length)
    axis_len = float(args.axis_length) if args.axis_length is not None else 0.5 * marker_len

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    # Camera intrinsics
    if args.camera and os.path.isfile(args.camera):
        camera_matrix, dist_coeffs = LoadCameraParams(args.camera)
    else:
        camera_matrix, dist_coeffs = GuessCameraParams(width, height, fov_deg=args.fov_deg)

    # ArUco dictionary and parameters
    dictionary = BuildDictionary(args.dict_name)
    try:
        parameters = cv2.aruco.DetectorParameters()
    except AttributeError:
        parameters = cv2.aruco.DetectorParameters_create()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output_path}")

    # CSV writer
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_index", "time_s", "found", "marker_id", "cx", "cy",
        "x", "y", "z", "A_deg", "B_deg", "C_deg", "rvec_x", "rvec_y", "rvec_z"
    ])

    trail_points = deque(maxlen=int(args.trail_max))
    frame_index = 0
    detections_ok = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = DetectMarkers(gray, dictionary, parameters)

        found_this_frame = False
        cx = cy = None
        tvec_log = (np.nan, np.nan, np.nan)
        A = B = C = np.nan
        rvec_log = (np.nan, np.nan, np.nan)

        if ids is not None and len(ids) > 0:
            for i, id_val in enumerate(ids.flatten()):
                if id_val == marker_id:
                    found_this_frame = True
                    detections_ok += 1
                    c = corners[i]

                    # Draw the detected marker border and ID
                    cv2.aruco.drawDetectedMarkers(frame, [c], np.array([[marker_id]], dtype=np.int32))

                    # Compute and store center
                    cx, cy = ComputeCenter(c)
                    trail_points.append((cx, cy))

                    # Pose estimate and axis
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([c], marker_len, camera_matrix, dist_coeffs)
                    if rvecs is not None and tvecs is not None and len(rvecs) > 0:
                        rvec = rvecs[0].reshape(3, 1)
                        tvec = tvecs[0].reshape(3, 1)

                        # Draw axes
                        DrawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_len)

                        # Log pose
                        R, _ = cv2.Rodrigues(rvec)
                        A, B, C = RotationMatrixToABC(R)
                        tvec_log = (float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0]))
                        rvec_log = (float(rvec[0, 0]), float(rvec[1, 0]), float(rvec[2, 0]))

                    # Draw center point with outline
                    if cx is not None and cy is not None:
                        cv2.circle(frame, (cx, cy), 6, (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)   # outline
                        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)  # blue
                    break  # Only track the first matching ID in this frame

        # Draw the trail path (black outline then blue line)
        if len(trail_points) >= 2:
            pts = np.array(trail_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 0), thickness=6, lineType=cv2.LINE_AA)
            cv2.polylines(frame, [pts], isClosed=False, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)

        # HUD text
        status_text = f"MarkerID:{marker_id}  Frame:{frame_index}  Detections:{detections_ok}"
        cv2.rectangle(frame, (10, 10), (10 + 560, 42), (0, 0, 0), thickness=-1)
        cv2.putText(frame, status_text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Write video frame
        writer.write(frame)

        # Write CSV row (one row per video frame)
        time_s = frame_index / fps
        csv_writer.writerow([
            frame_index,
            f"{time_s:.6f}",
            int(found_this_frame),
            marker_id if found_this_frame else "",
            cx if cx is not None else "",
            cy if cy is not None else "",
            f"{tvec_log[0]:.9f}" if not math.isnan(tvec_log[0]) else "",
            f"{tvec_log[1]:.9f}" if not math.isnan(tvec_log[1]) else "",
            f"{tvec_log[2]:.9f}" if not math.isnan(tvec_log[2]) else "",
            f"{A:.6f}" if not math.isnan(A) else "",
            f"{B:.6f}" if not math.isnan(B) else "",
            f"{C:.6f}" if not math.isnan(C) else "",
            f"{rvec_log[0]:.9f}" if not math.isnan(rvec_log[0]) else "",
            f"{rvec_log[1]:.9f}" if not math.isnan(rvec_log[1]) else "",
            f"{rvec_log[2]:.9f}" if not math.isnan(rvec_log[2]) else "",
        ])

        frame_index += 1

    cap.release()
    writer.release()
    csv_file.close()

    print(f"Done. Wrote video: {output_path}")
    print(f"Done. Wrote CSV:   {csv_path}")
    print("Notes:")
    print("- A,B,C are rotations about X,Y,Z (roll,pitch,yaw) in degrees, ZYX convention.")
    print("- For accurate metric poses, use --camera with real calibration.")
    print("- If axis overlay fails, ensure 'opencv-contrib-python' is installed and try upgrading OpenCV.")


if __name__ == "__main__":
    Main()
