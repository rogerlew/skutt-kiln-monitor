import cv2
import os
import numpy as np
import dotenv
import subprocess
import math
import json
import glob

from os.path import split as _split
from os.path import join as _join
from os.path import exists as _exists
# Debug level (0 = no debug info, 1 = console info, 2 = image output)
debug = 2

# 0. Set Working Directory
os.chdir("working")

#
# 1. Load username and password from .env file
#

dotenv.load_dotenv()
username = os.getenv('TAPO_USERNAME')
password = os.getenv('PASSWORD')
rtsp_ip = os.getenv('RTSP_IP')

#
# 2. Capture the RTSP stream using FFmpeg
#

# RTSP stream URL (adjust with your camera’s URL and credentials)
# RTSP camera is Tapo C110
rtsp_url = f"rtsp://{username}:{password}@{rtsp_ip}:554/stream1"

# Output filename pattern (e.g., frame_0001.jpg, frame_0002.jpg, ...)
output_pattern = "frame_%04d.jpg"

# Capture method (scene_change or frame_interval)
capture_method = "frame_interval" 

if capture_method == "frame_interval":
    # Frame interval (e.g., 1 frame every 2 seconds)
    frame_rate = 0.3  # Capture 0.5 fps (1 frame every 2 seconds)

    # FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-rtsp_transport", "tcp",    # Use TCP for stability
        "-i", rtsp_url,              # Input RTSP URL
        "-vf", f"fps={frame_rate}",  # Set the capture frame rate
        "-frames:v", "5",            # Capture 5 frames
        "-q:v", "2",                 # Image quality (lower number = better quality)
        output_pattern,              # Output file pattern
        "-y"                         # Overwrite files if they exist
    ]
elif capture_method == "scene_change":
    # when this fails the subprocess times out and no frames are captured
    tolerance = 0.025  # Scene change threshold (0.0 to 1.0)
    ffmpeg_command = [
        "ffmpeg",
        "-rtsp_transport", "tcp",  # Use TCP for stability
        "-i", rtsp_url,            # Input RTSP URL
        "-vf", f"select='gt(scene,{tolerance})',showinfo",  # Scene change threshold
        "-vsync", "vfr",           # Avoid duplicate frames
        "-frames:v", "5",          # Capture 5 frames
        output_pattern,            # Output file pattern
        "-y"                       # Overwrite existing files
    ]
else:
    print("Invalid capture method")
    exit(1)


print(' '.join(ffmpeg_command))

output_frames = glob.glob("frame_*.jpg")
for output_frame in output_frames:
    os.remove(output_frame)

try:
    # Run FFmpeg command
    subprocess.run(ffmpeg_command, check=True, timeout=20)
except subprocess.CalledProcessError as e:
    print(f"FFmpeg failed: {e}")

output_frames = glob.glob("frame_*.jpg")
assert len(output_frames) > 0, "ERROR: No frames captured"

if debug:
    print(f"Captured {len(output_frames)} frames\n\n")

#
# 3. Process the captured frames
#

#
# 3.1 Parameters for processing
#

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


#
# 3.2 Loop over the captured frames
#
H_box = None
rotate_angle = None
processed_frames = []
for k, output_frame in enumerate(output_frames):

    if debug:
        print(f"Processing frame {k}...")
    
    # Display the captured frame
    frame = cv2.imread(output_frame)
    if frame is None:
        print("  ERROR: Failed to load the captured frame.")

    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # normalize the gray
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    if debug > 1:
        cv2.imwrite(f"frame_{k+1:04d},01_gray.jpg", gray)

    #
    # 3.2.1 Detect markers
    if H_box is None:
        if debug:
            print("  Detecting markers...")
                
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if debug > 2:
            print("  Detected corners and markers:")
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

            if debug > 1:
                annotated = frame.copy()
                
                # Draw the fiducial’s corners
                for c in marker_corners.astype(int):
                    cv2.circle(annotated, tuple(c), 5, (0, 255, 0), -1)

                    # Draw box on original image for debugging
                    cv2.polylines(annotated, [box_corners_img.astype(np.int32)], True, (0,0,255), 1)
                    cv2.imwrite(f"frame_{k+1:04d},02_annotated_frame.jpg", annotated)

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
            
            if H_box is not None:
                print("  Homography identification successful")

            # Serialize homography to json
            homography = {
                "H": H_box.tolist(),
                "output_width": output_width,
                "output_height": output_height
            }
            with open("homography.json", "w") as f:
                json.dump(homography, f)

    #
    # 3.2.2 Homography Fallback
    if H_box is None:
        # this occurs in low-light conditions, so the camera could be setup
        # with good lighting and then after the lights go off the camera
        # can still retrieve the image from the last successful homography
        if debug:
            print("Marker not found")
        if  _exists("homography.json"):
            if debug:
                print("Using last successful homography")
            with open("homography.json", "r") as f:
                homography = json.load(f)
                H_box = np.array(homography["H"], dtype=np.float32)
                output_width = homography["output_width"]
                output_height = homography["output_height"]
                
    assert H_box is not None, "ERROR: Cannot determine homography"
        
        
    #
    # 3.2.3 Image rectification
    if debug:
        print("  Rectifying image...")
        
    # Custom weighted grayscale conversion without clipping
    # ITU-R BT.601 has default weights of Red: 0.299, Green: 0.587, Blue 0.114
    weighted_gray = (
        0.299 * 0.8 * frame[:, :, 2] +  # Red channel
        0.587 * frame[:, :, 1] +   # Green channel (emphasized)
        0.114 * 0.5 * frame[:, :, 0]     # Blue channel (subtracted)
    )

    # Calculate the raw min and max values
    min_val = np.min(weighted_gray)
    max_val = np.max(weighted_gray)

    # Normalize to range [0, 255] using the raw min/max
    weighted_gray = ((weighted_gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Display results
    if debug:
        print(f"  Grayscale conversion,  Raw Min Value: {min_val}, Raw Max Value: {max_val}")
    
    rectified_box = cv2.warpPerspective(weighted_gray, H_box, (output_width, output_height))


    if debug > 1:
        cv2.imwrite(f"frame_{k+1:04d},03_rectified.jpg", rectified_box)

    #
    # 3.2.4 Detect lines in the rectified image to correct small rotation errors
    if rotate_angle is None:
        if debug:
            print("  Detecting lines to determine rotation...")
            
        # The Canny detection fails to find the edges of the segmented display with the custom grayscale conversion
        _rectified_box = cv2.warpPerspective(gray, H_box, (output_width, output_height))
            
        edges = cv2.Canny(_rectified_box, 50, 150, apertureSize=3)

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
            
            # serialize rotate_angle to json
            with open("rotate_angle.json", "w") as f:
                json.dump({"angle": rotate_angle}, f)

            if debug > 1:
                rect_annotated = _rectified_box.copy()
                # make color
                rect_annotated = cv2.cvtColor(rect_annotated, cv2.COLOR_GRAY2BGR)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(rect_annotated, (x1, y1), (x2, y2), (0,0,255), 1)
                    
                # Draw the best line
                x1, y1, x2, y2 = best_line
                cv2.line(rect_annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(rect_annotated, f"Angle: {rotate_angle:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(f"frame_{k+1:04d},04_lines.jpg", rect_annotated)

    #
    # 3.2.5 Fallback for rotation angle
    if rotate_angle is None:
        if debug:
            print("  Angle detection failed")
        if _exists("rotate_angle.json"):
            if debug:
                print("Using last successful angle")
            with open("rotate_angle.json", "r") as f:
                rotate_angle = float(json.load(f)["angle"])
                
    if rotate_angle is None:
        if debug:
            print("  Defaulting to 0 rotation angle")
        rotate_angle = 0.0
        
    if abs(rotate_angle) > 5.0:
        if debug:
            print(f"  WARNING: Detected rotation angle of {rotate_angle:.2f} degrees, falling back to 0.0")
        rotate_angle = 0.0

    #
    # 3.2.6 Rotate image
    if debug:
        print(f"  Rotating image by {rotate_angle:.2f} degrees")
        
    # Rotate image around its center
    (h, w) = rectified_box.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    corrected_box = cv2.warpAffine(rectified_box, M, (w, h), flags=cv2.INTER_LINEAR)

    if debug > 1:
        print(f"  Corrected image saved as frame_{k+1:04d},05_corrected.jpg")
        cv2.imwrite(f"frame_{k+1:04d},05_corrected.jpg", corrected_box)
        rectified_box = corrected_box
        
    # 3.2.7 Append the processed frame
    processed_frames.append(corrected_box)
    
    if debug:
        print(f"")
        
#
# 3.3 Read the processed frames
#

char_origins = [
    (52, 58), # (131, 202)
    (178, 58),
    (304, 58),
    (434, 58)
]

char_width = 135 - 52
char_height = 202 - 58

segment_offsets = [
    (100-52,  68-58), # 0
    (262-178, 98-58),  # 1
    (256-178, 158-58), # 2
    (91-52,   192-58), # 3
    (59-52,   158-58), # 4
    (63-52,   98-58), # 5
    (207-178, 129-58),  # 6
    (241-178, 129-58),  # 7
    (476-434, 98-58),  # 8
    (474-434, 158-58),  # 9
]

segment_offsets = [(int(x) / char_width, int(y) / char_height) for x, y in segment_offsets]

background_offsets = [
    (38-52, 44-58),
    (100-52, 44-58),
    (152-52, 44-58),
    (38-52, 214-58),
    (100-52, 214-58),
    (152-52, 214-58),
    (36-52, 98-58),
    (35-52, 159-58),
    (152-52, 98-58),
    (152-52, 159-58),
]

background_offsets = [(int(x) / char_width, int(y) / char_height) for x, y in background_offsets]

if debug:
    print("Reading processed frames...")

for k, image in enumerate(processed_frames):
    if debug:
        print(f"  Reading frame {k+1}...")
    
    if debug > 1:
        annotated = image.copy()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
        for char_origin in char_origins:
            ul_x, ul_y = char_origin
            lr_x = ul_x + char_width
            lr_y = ul_y + char_height
            
            box_corners = np.array([
                [ul_x, ul_y],
                [lr_x, ul_y],
                [lr_x, lr_y],
                [ul_x, lr_y]
            ], dtype=np.int32)
                
            cv2.polylines(annotated, [box_corners], True, (0, 0, 255), 1)
            
            for x, y in segment_offsets:
                x = int(x * char_width + ul_x)
                y = int(y * char_height + ul_y)
                cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
                
            for x, y in background_offsets:
                x = int(x * char_width + ul_x)
                y = int(y * char_height + ul_y)
                cv2.circle(annotated, (x, y), 5, (255, 0, 0), -1)
        
    cv2.imwrite(f"frame_{k+1:04d},06_char_segmentation.jpg", annotated)
    
