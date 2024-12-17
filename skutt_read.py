import cv2
import os
import numpy as np
import dotenv
import subprocess
import math

# load username and password from .env file
dotenv.load_dotenv()
username = os.getenv('TAPO_USERNAME')
password = os.getenv('PASSWORD')
rtsp_ip = os.getenv('RTSP_IP')

# RTSP stream URL (adjust with your camera’s URL and credentials)
# RTSP camera is Tapo C110
rtsp_url = f"rtsp://{username}:{password}@{rtsp_ip}:554/stream1"

output_frame = "00_frame.jpg"

# Run FFmpeg command to grab a single frame
ffmpeg_command = [
    "ffmpeg", 
    "-rtsp_transport", "tcp",  # Use TCP for stability
    "-i", rtsp_url,            # Input RTSP URL
    "-frames:v", "1",          # Capture only one frame
    "-q:v", "2",               # Image quality (lower number is higher quality)
    output_frame,              # Output file
    "-y"                       # Overwrite output file if it exists
]

print(' '.join(ffmpeg_command))

frame = None

try:
    # Run FFmpeg command
    subprocess.run(ffmpeg_command, check=True)
    print(f"Frame saved as {output_frame}")
    
    # Display the captured frame
    frame = cv2.imread(output_frame)
    if frame is None:
        print("Failed to load the captured frame.")

except subprocess.CalledProcessError as e:
    print(f"FFmpeg failed: {e}")

debug = 1

# Load the ArUco dictionary and detector parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Define the marker ID we are looking for
target_id = 8

# Define bounding box in normalized units relative to marker size
# (0,0) top-left of marker, (1,1) bottom-right of marker
UL = (1.07, 0.3)  # upper-left corner of bounding box
LR = (2.2, 0.8)  # lower-right corner of bounding box
scale = 500  # pixels per unit

# Points defining the normalized marker coordinate system
# We assume the marker corners map to these points:
# top-left: (0,0)
# top-right: (1,0)
# bottom-right: (1,1)
# bottom-left: (0,1)
marker_corners_normalized = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
], dtype=np.float32)

# Convert to grayscale for ArUco detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# normalize the gray
gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

if debug:
    cv2.imwrite("01_gray.jpg", gray)

# Detect markers
corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

print(corners, ids)

if ids is not None and target_id in ids:
    idx = np.where(ids == target_id)[0][0]
    marker_corners = corners[idx][0]  # shape (4,2), order: TL, TR, BR, BL

    # Normalized marker coordinates (unit square)
    marker_corners_normalized = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    # Compute homography from image -> normalized marker coords
    H, status = cv2.findHomography(marker_corners, marker_corners_normalized)

    # Define box in normalized marker coordinates
    box_corners_normalized = np.array([
        [UL[0], UL[1]],    # UL
        [LR[0], UL[1]],    # UR
        [LR[0], LR[1]],    # LR
        [UL[0], LR[1]]     # LL
    ], dtype=np.float32)

    # Inverse homography to get box corners in image coordinates
    H_inv = np.linalg.inv(H)
    ones = np.ones((4,1), dtype=np.float32)
    box_hom = np.hstack([box_corners_normalized, ones])
    box_img_hom = (H_inv @ box_hom.T).T
    box_corners_img = (box_img_hom[:, :2] / box_img_hom[:, 2:3]).astype(np.float32)

    if debug:
        annotated = frame.copy()
        
        # Draw the fiducial’s corners
        for c in marker_corners.astype(int):
            cv2.circle(annotated, tuple(c), 5, (0, 255, 0), -1)

            # Draw box on original image for debugging
            cv2.polylines(annotated, [box_corners_img.astype(np.int32)], True, (0,0,255), 1)
            cv2.imwrite("03_annotated_frame.jpg", annotated)

    # Compute bounding box dimensions in normalized space
    w = LR[0] - UL[0]
    h = LR[1] - UL[1]
    output_width = int(w * scale)
    output_height = int(h * scale)

    # Destination box for the warp
    dst_box = np.array([
        [0.0, 0.0],
        [output_width - 1, 0.0],
        [output_width - 1, output_height - 1],
        [0.0, output_height - 1]
    ], dtype=np.float32)

    # Compute new homography from original box to destination box
    H_box, status = cv2.findHomography(box_corners_img, dst_box)

    # Warp the frame to get a rectified view of just the bounding box
    rectified_box = cv2.warpPerspective(gray, H_box, (output_width, output_height))
    
    # normalize the rectified_box
    rectified_box = cv2.normalize(rectified_box, None, 0, 255, cv2.NORM_MINMAX)

    if debug:
        cv2.imwrite("04_rectified.jpg", rectified_box)

    blur = cv2.GaussianBlur(rectified_box, (3,3), 0)
    edges = cv2.Canny(rectified_box, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=30,
                        minLineLength=150,
                        maxLineGap=50)  

    if lines is not None:
        # Find the best line as the longest line
        best_line = None
        longest_line = 0.0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            angle = math.degrees(math.atan2(dy, dx))
            length = math.sqrt(dx*dx + dy*dy)
            if length > longest_line:
                longest_line = length
                rotate_angle = angle
                best_line = (x1, y1, x2, y2)

        if debug:
            rect_annotated = rectified_box.copy()
            # make color
            rect_annotated = cv2.cvtColor(rect_annotated, cv2.COLOR_GRAY2BGR)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(rect_annotated, (x1, y1), (x2, y2), (0,0,255), 1)
                
            # Draw the best line
            x1, y1, x2, y2 = best_line
            cv2.line(rect_annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            
            # write the angle on the image
            cv2.putText(rect_annotated, f"Angle: {rotate_angle:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.imwrite("05_lines.jpg", rect_annotated)

        # Rotate image around its center
        (h, w) = rectified_box.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
        corrected_box = cv2.warpAffine(rectified_box, M, (w, h), flags=cv2.INTER_LINEAR)

        if debug:
            cv2.imwrite("06_corrected.jpg", corrected_box)
            rectified_box = corrected_box
            
    