# Author: Roger Lew
# License: BSD-3-Clause

import cv2
import os
import sys
import numpy as np
import random
import dotenv
import subprocess
import math
import json
import glob
from datetime import datetime

from os.path import split as _split
from os.path import join as _join
from os.path import exists as _exists

import paho.mqtt.client as mqtt
import argparse

# Developer notes:
#
# This purposefully uses a script approach because it is easier to debug and not get lost.
# With these sort of CV scripts, there is alot of trial and error and complex program flows
# end up causing problems if it isn't doing what you actually think it is doing.
#
# I also knew at the outset I would be deploying it as crontab on Linux. So the scripted
# approach is more robust for that purpose. 

# 1.1 Configure debug

# Debug level (0 = no debug info, 1 = console info, 2 = image output)
debug = 2

parser = argparse.ArgumentParser(description="skutt_read.py --no_debug flag")
parser.add_argument(
    "--no_debug",
    action="store_true",
    help="Disable debug mode"
)
args = parser.parse_args()
if args.no_debug:
    debug = 0


thisdir = os.path.dirname(os.path.abspath(__file__))

# 1.2 Set Working Directory

workdir = '/ramdisk'

if not _exists(workdir):
    workdir = 'working'
    os.makedirs(workdir, exist_ok=True)

os.chdir(workdir)

# 1.3 Configure logs

error_log_fn = '/var/log/skutt-monitor/error.log'
if not _exists(_split(error_log_fn)[0]):
    error_log_fn = 'error.log'
    
run_log_fn = '/var/log/skutt-monitor/run.log'
if not _exists(_split(run_log_fn)[0]):
    run_log_fn = 'run.log'
    
# 1.4 Load the template image

template_fn = _join(thisdir, 'templates/template.jpg')

if not _exists(template_fn):
    sys.stderr.write("ERROR: Template file not found\n")
        
    with open(error_log_fn, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] ERROR: Template file not found\n")
        
    exit(1)

segment_mask_fn = _join(thisdir, 'templates/segment_mask.png')

if not _exists(segment_mask_fn):
    sys.stderr.write("ERROR: Segment Mask file not found\n")
        
    with open(error_log_fn, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] ERROR: Segment Mask file not found\n")
        
    exit(1)

#
# 1.5 Load username and password from .env file
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

# The Skutt kiln cycles through state, temperature, and time, we collect 7 frames
# so that we can be sure to observe state -> temperature event
n_frames = 7  

if capture_method == "frame_interval":
    # Frame interval (e.g., 1 frame every 2 seconds)
    frame_rate = 0.5  # Capture 0.5 fps (1 frame every 2 seconds)

    # FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-rtsp_transport", "tcp",    # Use TCP for stability
        "-i", rtsp_url,              # Input RTSP URL
        "-vf", f"fps={frame_rate}",  # Set the capture frame rate
        "-frames:v", f"{n_frames}",  # Capture 5 frames
        "-q:v", "2",                 # Image quality (lower number = better quality)
        output_pattern,              # Output file pattern
        "-y"                         # Overwrite files if they exist
    ]
elif capture_method == "scene_change":
    # when this fails the subprocess times out and no frames are captured
    tolerance = 0.025  # Scene change threshold (0.0 to 1.0)
    ffmpeg_command = [
        "ffmpeg",
        "-rtsp_transport", "tcp",   # Use TCP for stability
        "-i", rtsp_url,             # Input RTSP URL
        "-vf", f"select='gt(scene,{tolerance})',showinfo",  # Scene change threshold
        "-vsync", "vfr",            # Avoid duplicate frames
        "-frames:v", f"{n_frames}", # Capture 5 frames
        output_pattern,             # Output file pattern
        "-y"                        # Overwrite existing files
    ]
else:
    sys.stderr.write("ERROR: Invalid capture method\n")
        
    with open(error_log_fn, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] ERROR: Invalid capture method\n")
        
    exit(1)

if debug > 4:
    print(' '.join(ffmpeg_command))

#
# 2.1 Remove existing frames
output_frames = glob.glob("frame_*.jpg")
for output_frame in output_frames:
    os.remove(output_frame)

#
# 2.2 Capture the RTSP stream
result = subprocess.run(
    ffmpeg_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=4 * n_frames,
    text=True  # Captures output as string
)

if result.returncode != 0:
    sys.stderr.write(f"ERROR: FFmpeg failed with return code {result.returncode}: {result.stderr}\n")
    
    with open(error_log_fn, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] ERROR: FFmpeg failed with return code {result.returncode}: {result.stderr}\n")
        
    exit(1)
    
    
# Check if frames were captured
output_frames = glob.glob("frame_*.jpg")

if debug:
    print(f"Captured {len(output_frames)} frames\n\n")

if len(output_frames) == 0:
    sys.stderr.write("ERROR: No frames captured\n")
        
    with open(error_log_fn, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] ERROR: No frames captured\n")
        
    exit(1)
#
# 3. Process the captured frames
#

#
# 3.1 Parameters for processing.
#
#
# The goal here is to get the ROI for the segmented display.
# it does this by using a 2" x 2" aruco marker to determine the
# homography between the camera and the marker. Then the image is
# rectified using the pose of the marker. Then the image is further refined
# by using a template matching algorithm to zero in on the segmented display.
# 
# The segment reading uses point locations to determine whether the segments
# are active or not, and the locaitons have to be pretty precise.

# Load the ArUco dictionary and detector parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Define the marker ID we are looking for
target_id = 8

# Define bounding box in normalized units relative to marker size
# (0,0) top-left of marker, (1,1) bottom-right of marker
#
#
# The bounding box extends above the marker and all the way across the front panel of the control box
# The bounding box also captures part of the keypad below the segmented display. The goal is
# to capture features for the fine homography that will be used to correct the image.
UL = (-0.25, -1.0)  # upper-left corner of bounding box
LR = (3.5, 1.5)  # lower-right corner of bounding box
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
# The frames are assumed to have the same perspective and we only identify the homography for the first frame
# for the remaining frames we use the homography from the first frame to rectify the image
H_box, H_fine = None, None
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
                if debug:
                    print("  Homography identification successful")

            # Serialize homography to json
            homography = {
                "H": H_box.tolist(),
                "output_width": output_width,
                "output_height": output_height
            }
            with open("H_box.json", "w") as f:
                json.dump(homography, f)

    #
    # 3.2.2 Homography Fallback
    if H_box is None:
        # this occurs in low-light conditions, so the camera could be setup
        # with good lighting and then after the lights go off the camera
        # can still retrieve the image from the last successful homography
        if debug:
            print("Marker not found")
        if  _exists("H_box.json"):
            if debug:
                print("Using last successful homography")
            with open("H_box.json", "r") as f:
                homography = json.load(f)
                H_box = np.array(homography["H"], dtype=np.float32)
                output_width = homography["output_width"]
                output_height = homography["output_height"]
                
    if H_box is None:
        sys.stderr.write("ERROR: Cannot determine homography\n")
            
        with open(error_log_fn, 'a') as f:
            f.write(f"[{datetime.now().isoformat()}] ERROR: Cannot determine homography\n")
            
        exit(1)
    
    #
    # 3.2.3 Image rectification
    if debug:
        print("  Rectifying image...")
        
    # Custom weighted grayscale conversion without clipping
    # ITU-R BT.601 has default weights of Red: 0.299, Green: 0.587, Blue 0.114
    weighted_gray = (
        0.299 * 0.8 * frame[:, :, 2] +  # Red channel
        0.587 * frame[:, :, 1] +        # Green channel (emphasized)
        0.114 * 0.5 * frame[:, :, 0]    # Blue channel (subtracted)
    )

    # Calculate the raw min and max values
    min_val = np.min(weighted_gray)
    max_val = np.max(weighted_gray)

    # Normalize to range [0, 255] using the raw min/max
    weighted_gray = ((weighted_gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Display results
    if debug:
        print(f"  Grayscale conversion,  Raw Min Value: {min_val}, Raw Max Value: {max_val}")
    
    # Rectify the image
    rectified_box = cv2.warpPerspective(weighted_gray, H_box, (output_width, output_height), flags=cv2.INTER_NEAREST)

    if debug > 1:
        cv2.imwrite(f"frame_{k+1:04d},03_rectified.jpg", rectified_box)

    if H_fine is None:
        if debug:
            print("  Detecting features for fine homography...")
            
        template = cv2.imread(template_fn, cv2.IMREAD_GRAYSCALE)
        
        detector = cv2.AKAZE_create()
        kp1, des1 = detector.detectAndCompute(rectified_box, None)
        kp2, des2 = detector.detectAndCompute(template, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)
                
        if debug:
            print(f"  Found {len(good_matches)} good matches")
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
        if len(good_matches) > 4:
            H_fine, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
        if debug > 1:
            rect_annotated = rectified_box.copy()
            # make color
            rect_annotated = cv2.cvtColor(rect_annotated, cv2.COLOR_GRAY2BGR)
            
            # Draw the matches
            rect_annotated = cv2.drawMatches(rectified_box, kp1, template, kp2, good_matches, rect_annotated, flags=2)
                
            cv2.putText(rect_annotated, f"Matches: {len(good_matches)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(f"frame_{k+1:04d},04_feature_matching.jpg", rect_annotated)

        if H_fine is not None:
            if debug:
                print("  Homography identification successful")

            # Serialize homography to json
            homography = {
                "H": H_fine.tolist(),
                "output_width": output_width,
                "output_height": output_height
            }
            with open("H_fine.json", "w") as f:
                json.dump(homography, f)
        else:
            if debug:
                print("  WARNING: Fine Homography identification failed")

    #
    # 3.2.2 Homography Fallback
    if H_fine is None:
        # this occurs in low-light conditions, so the camera could be setup
        # with good lighting and then after the lights go off the camera
        # can still retrieve the image from the last successful homography
        if  _exists("H_fine.json"):
            if debug:
                print("Using last successful homography")
            with open("H_fine.json", "r") as f:
                homography = json.load(f)
                H_fine = np.array(homography["H"], dtype=np.float32)
                output_width = homography["output_width"]
                output_height = homography["output_height"]
                
    if H_fine is None:
        sys.stderr.write("ERROR: Cannot determine homography\n")
            
        with open(error_log_fn, 'a') as f:
            f.write(f"[{datetime.now().isoformat()}] ERROR: Cannot determine homography\n")
            
        exit(1)

    h_t, w_t = template.shape[:2]
    corrected_box = cv2.warpPerspective(rectified_box, H_fine, (w_t, h_t), flags=cv2.INTER_NEAREST)

    if debug > 1:
        print(f"  Corrected image saved as frame_{k+1:04d},05_corrected.jpg")
        cv2.imwrite(f"frame_{k+1:04d},05_corrected.jpg", corrected_box)
        
    # WARNING: this isn't in fiducial units, its in pixels. so if scale is changed
    # this will need to be updated
    #
    # This defines the ROI for the segmented display
    # The cropped image should be 565 x 250 pixels for the other parameters to work
    cropped = corrected_box[653:653+250, 664:664+565]
    
    if debug > 1:
        cv2.imwrite(f"frame_{k+1:04d},06_cropped.jpg", cropped)
    
    # 3.2.7 Append the processed frame
    processed_frames.append(cropped)
    
    if debug:
        print(f"")
 
if len(processed_frames) == 0:
    sys.stderr.write("ERROR: No processed frames\n")
        
    with open(error_log_fn, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] ERROR: No processed frames\n")
        
    exit(1)
              

#
# 4. Read the processed frames
#
# this is loosely based on the approach by https://github.com/scottmudge/SegoDec/


# these specfies the origin of the 4 characters in the display in pixels coordinates
# the origin is the top-left corner of the character
char_origins = [
    (52-8, 58-8), # (131, 202)
    (178-8, 58-8),
    (304-5, 58-8),
    (434-8, 58-8)
]

# these are used to hit test contours for the precise character origin setting
char_width = 135 - 30
char_height = 202 - 57


#
# 4.0 build a composite frame to finetune the character origins
#
# composite frames together to help with identifying location of segments
composite_frame = processed_frames[0].copy()
for frame in processed_frames[1:]:
    composite_frame = np.maximum(composite_frame, frame)
composite_frame = cv2.normalize(composite_frame, None, 0, 255, cv2.NORM_MINMAX)
composite_frame = composite_frame.astype(np.uint8)

if debug > 1:
    cv2.imwrite("roi_composite_frame.jpg", composite_frame)

_, thresh = cv2.threshold(composite_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

def bounds_intersect(bbox1, bbox2):
    """
    Determine if two bounding boxes intersect.

    Parameters:
    bbox1, bbox2: Tuples representing the bounding boxes in the format (x, y, w, h)

    Returns:
    True if the bounding boxes intersect, False otherwise
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the bottom-right coordinates of both bounding boxes
    x1_br, y1_br = x1 + w1, y1 + h1
    x2_br, y2_br = x2 + w2, y2 + h2

    # Check for no overlap conditions
    if x1_br <= x2 or x2_br <= x1:
        return False
    if y1_br <= y2 or y2_br <= y1:
        return False

    # If none of the no overlap conditions are met, the boxes intersect
    return True

# filter out bounding boxes that intersect with more than one character_box
filtered_bounding_boxes = []

for bbox in bounding_boxes:
    num_intersections = 0
    for ul_x, ul_y in char_origins:
        if bounds_intersect(bbox, (ul_x, ul_y, char_width, char_height)):
            num_intersections += 1
            
    if num_intersections == 1:
        filtered_bounding_boxes.append(bbox)

_revised_char_origins = []

for k, (ul_x, ul_y) in enumerate(char_origins):
    lr_x = ul_x + char_width
    lr_y = ul_y + char_height
    
    intersections = []
    for x, y, w, h in filtered_bounding_boxes:
        if bounds_intersect((ul_x, ul_y, char_width, char_height), (x, y, w, h)):
            intersections.append((x, y, w, h))
            
    if len(intersections) == 0:
        if debug:
            print(f"WARNING: No intersection found for character {k+1} at {ul_x, ul_y}")
            
        _revised_char_origins.append((ul_x, ul_y))
        
    else:
        # build bounding box of the intersections
        x, y, w, h = intersections[0]
        for x1, y1, w1, h1 in intersections[1:]:
            x = min(x, x1)
            y = min(y, y1)
            w = max(w, x1 + w1) - x
            h = max(h, y1 + h1) - y
            
        distance = math.sqrt((x - ul_x)**2 + (y - ul_y)**2)
        if distance < 20:
            if debug: 
                print(f"Revising character origin for character {k+1} at {ul_x, ul_y} to {x, y}")
            _revised_char_origins.append((x, y))
        else:
            if debug:
                print(f"WARNING: Distance too large for revising character {k+1} origin at {ul_x, ul_y} to {x, y}")
            _revised_char_origins.append((ul_x, ul_y))

char_origins = _revised_char_origins

if debug > 1:
    output_image = cv2.cvtColor(composite_frame, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization
    for (x, y, w, h) in bounding_boxes:
        if w > 10 and h > 20:  # Filter small artifacts
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("roi_bounding_boxes.jpg", output_image)

#
# Segment and character defintions
#

"""
Segment Mask - Segments are index like so:

    ####          0 0       # # #  
   #    #       5  8  1    #10 12#  
   #    #       5  8  1    #     #  
    ####   ==>    6 7       # # #  
   #    #       4  9  2    #     #  
   #    #       4  9  2    #11 13#  
    ####          3 3       # # #  14

    The segment mask indicates which segments are active
    during each displayed number.
"""

# these define the segment for each character
# the alphanumeric characters isn't comprehensive, but includes characters
# that are routinely displayed on the kiln.
#
# A better way to do this would be to use more standardized segment mapping
# and predefined dictionaries. e.g. https://github.com/dmadison/LED-Segment-ASCII/tree/master
#
# ... but I got to far before I realized it and Skutt some novel choices with the
# font selection and I only gave myself a 24 hour period to do this. 

segment_definitions = {
        # 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
    "0": (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "1": (0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "2": (1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    "3": (1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    "4": (0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    "5": (1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    "6": (1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    "7": (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "8": (1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    "9": (1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    "C": (1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "P": (1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    "L": (0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "T": (1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    "S": (1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    "E": (1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    "W": (0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0),
    "d": (0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
    "I": (1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    "R": (1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0),
    "H": (0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    "N": (0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0),
   "--": (0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    " ": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ".": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
}

_segment_mask_origins = [
    (422, 49),  # 0
    (422, 49),  # 1
    (422, 49),  # 2
    (422, 49),  # 3
    (422, 49),  # 4
    (422, 49),  # 5
    (422, 49),  # 6
    (422, 49),  # 7
    (422, 49),  # 8
    (422, 49),  # 9
    (422, 49),  # 10
    (422, 49),  # 11
    (422, 49),  # 12
    (422, 49),  # 13
    (165, 48),  # 14
]

_segment_mask = cv2.imread(segment_mask_fn)

# we need to build a segment_offset_mask that can be used to slice the values in the image
segment_offsets = []

for i, (x, y) in enumerate(_segment_mask_origins):
    # find indices of _segment_mask where blue = 0, red = 0, and green = 230 + i
    mask = (_segment_mask[:, :, 0] == 0) & (_segment_mask[:, :, 1] == 230 + i) & (_segment_mask[:, :, 2] == 0)
    indices = np.argwhere(mask)
    y_indx, x_indx = indices.T
    
    assert len(y_indx) > 0, f"No segment found for segment {i}, bad segment mask file"
    
    y_indx -= y
    x_indx -= x
    segment_offsets.append((x_indx, y_indx))


# These define the background sampling points around each segment.
# The idea is that we know these are background and can use there values
# to constrast active segments.
mask = (_segment_mask[:, :, 0] == 255) & (_segment_mask[:, :, 1] == 0) & (_segment_mask[:, :, 2] == 0)
indices = np.argwhere(mask)
y_indx, x_indx = indices.T
y_indx -= 50 # origin of first character
x_indx -= 32
background_offsets = (x_indx, y_indx)

#
# Some Helper functions to avoid heavy nesting
#

def otsu_like_threshold(background_pts, segment_pts):
    """
    Calculate a threshold value that separates background points from segment points
    using a method similar to Otsu's thresholding.
    Parameters:
    background_pts (array-like): Array of background points.
    segment_pts (array-like): Array of segment points.
    Returns:
    int: The calculated threshold value. If the calculated threshold is less than 100,
         the function returns 100.
    """
    
    # Combine all sample points
    all_pts = np.concatenate([background_pts, segment_pts])
    all_pts.sort()

    n = len(all_pts)
    nb = len(background_pts)  # number of background points

    best_threshold = None
    min_value = float('inf')

    # Start iteration from nb+1 so that class 1 includes all background points
    # and class 2 contains only segment (or mostly segment) points.
    for i in range(nb+1, n-2):
        # Split all_pts into two groups: [0:i], [i:n]
        n2 = n - i
        n1 = n - n2
        
        x1 = all_pts[:i]
        var1 = np.var(x1)
        
        x2 = all_pts[i:]
        var2 = np.var(x2)
        
        value = var2 * n2 + var1 * n1
        
        if debug > 4:
            print(x2, var1, n1,  var2, n2, value)

        if value < min_value:
            min_value = value
            best_threshold = x2[0]-1 

    if best_threshold < 100:
        return 100
    
    return best_threshold


def read_segments(image: np.ndarray, origin: tuple, method='point_otsu') -> list:
    """
    Read the segments of a character from an image.
    
    method: 'confidence_interval' 'point_otsu' or 'otsu' or 'ensemble'
    """
    global debug, segment_offsets, background_offsets
    
    if debug:
        print("read_segments", origin)
        
    if not method in ['confidence_interval', 'point_otsu', 'otsu', 'ensemble']:
        sys.stderr.write("ERROR: Invalid method\n")
        
        with open(error_log_fn, 'a') as f:
            f.write(f"[{datetime.now().isoformat()}] ERROR: Invalid method\n")
            
        exit(1)
    
    origin_x, origin_y = origin
    
    background_pts = []
    
    x_offsets, y_offsets = background_offsets
    _x = origin_x + x_offsets
    _y = origin_y + y_offsets
    background_pts = image[_y, _x].astype(int).tolist()
    background_pts = random.sample(background_pts, 14)
    
    segment_pts = []
    for x_offsets, y_offsets in segment_offsets:
        _x = origin_x + x_offsets
        _y = origin_y + y_offsets
        segment_pts.append(int(np.median(image[_y, _x])))
        
    if debug:
        print("  Background Points", background_pts)
        print("  Segment Points:", segment_pts)
    
    ci_segments = None
    if method == 'confidence_interval' or method == 'ensemble':
        mu = np.mean(background_pts)
        sigma = np.std(background_pts)
        ci = 4 * sigma
        threshold = mu + ci
        
        ci_segments = [int(value > threshold) for value in segment_pts]
    
        if debug:
            print("  CI Segmentation")
            print("    Mean:", mu)
            print("    Std Dev:", sigma)
            print("    Threshold:", threshold)
            print("    Segments:", ci_segments)
            
    otsu_segments = None
    if method == 'otsu' or method == 'ensemble':
        _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        otsu_segments =[]
        for x_offset, y_offset in segment_offsets:
            _x = origin_x + x_offset
            _y = origin_y + y_offset
            value = otsu_thresh[_y, _x]
            otsu_segments.append(int(value / 255))
            
        if debug:
            print("  Otsu Segmentation")
            print("    Segments:", otsu_segments)
        
    otsu_pt_segments = None
    if method == 'point_otsu' or method == 'ensemble':
        o2_threshold_val = otsu_like_threshold(background_pts, segment_pts)
        otsu_pt_segments = [int(value > o2_threshold_val) for value in segment_pts]
        
        if debug:
            print("  Otsu Point Segmentation")
            print("    Threshold:", o2_threshold_val)
            print("    Segments:", otsu_pt_segments)
            
    if method == 'confidence_interval':
        return ci_segments
    elif method == 'otsu':
        return otsu_segments
    elif method == 'point_otsu':
        return otsu_pt_segments
    elif method == 'ensemble':
        return [int((a + b + c) > 1) for a, b, c in zip(ci_segments, otsu_segments, otsu_pt_segments)]


def read_char(image: np.ndarray, origin: tuple, error_tolerance=2) -> str:
    """
    Read a character from an image.
    """
    global segment_definitions
    
    segments = read_segments(image, origin)
    
    if (segments[8] == 1 and segments[9] == 1):
        if debug and sum(segments[10:14]) > 0:
            print("  WARNING: setting segments 10, 11, 12, 13 to 0 due to 8 and 9 being active")
            
        segments[10] = 0
        segments[11] = 0
        segments[12] = 0
        segments[13] = 0
    
    ret = ''
    
    n = len(segments[:-1])
    match_counts = {}
    for char, mask in segment_definitions.items():
        # need to ignore the period segment for now or it messes up the matching counts
        if char == '.':
            continue
        match_counts[char] = n - sum(m == s for m, s in zip(mask[:-1], segments[:-1]))
            
    # Find the best match and append to ret is within the error tolerance
    best_match = min(match_counts, key=match_counts.get)
    
    if debug:
        print(f"    Best Match: {best_match}")
        print(f"    Match Error: {match_counts[best_match]}")
    
    if match_counts[best_match] <= error_tolerance:
        ret += best_match
            
    if segments[-1] == 1:
        ret += '.'
        
    if len(ret) == 0:
        ret = " "

    return ret


def read_display(image: np.ndarray) -> str:
    """
    Reads all th echaracters from an image.
    """
    global char_origins
    output = ""
    for i, char_origin in enumerate(char_origins):
        if debug:
            print(f"Reading char {i+1} @ {char_origin}")
        output += read_char(image, char_origin)
        
    return output

# 4.1 LFG

if debug:
    print("Reading processed frames...")

displays = [] # holds the characters from each of the frames
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
            
            for xs, ys in segment_offsets:
                annotated[ul_y+ys, ul_x+xs] = (0, 255, 0)
                
            x_indx, y_indx = background_offsets
            annotated[ul_y+y_indx, ul_x+x_indx] = (255, 0, 0)
            
        cv2.imwrite(f"frame_{k+1:04d},06_char_segmentation.jpg", annotated)
    
    display = read_display(image)
    
    if debug:
        print(f"  Display: {display}\n")
        
    displays.append(display)
    
if len(displays) == 0:
    sys.stderr.write("ERROR: No displays\n")
        
    with open(error_log_fn, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] ERROR: No displays\n")
        
    exit(1)
    
#
# 4.2 Now that we have the displays, we can extract the temperature, time, and state
#
def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

    
def skutt_time(x):
    """
    Returns the time in HH:MM format if the input is identified as a time string,
    otherwise returns None.
    """
    if '.' in x:
        _x = x.split('.')
        if len(_x) != 2:
            return None
        
        _x0, _x1 = _x
        try:
            hours = int(_x0)
            minutes = int(_x1)
            return f'{hours:02d}:{minutes:02d}'
        except ValueError:
            return None
    
    return None

def skutt_temp(x, last_was_state):
    """
    Returns the temperature in DEG F if the input is identified as a temperature string,
    otherwise returns None.
    """
    try:
        temp = float(x)
        if temp > 60.0 and temp < 2500.0 and last_was_state:
            return temp
    except ValueError:
        return None
    
    return None
            
            
def skutt_state(x):
    """
    Returns the state if the input is identified as a state string,
    otherwise returns None.
    
    Not fully implemented.
    """
    if x.startswith('CPL'):
        return 'Complete'
    if 'CP' in x:
        return 'Complete'
    if 'PL' in x:
        return 'Complete'
    if 'LT' in x:
        return 'Complete'

    return None

temp, time, state = None, None, None
last_was_state = None

if debug:
    print("Frame Semantics:")
    
for i, display in enumerate(displays):
    if debug:
        print(f"  Frame {i+1}: {display}")
        
    _temp = skutt_temp(display, last_was_state)
    _time = skutt_time(display)
    _state = skutt_state(display)
            
    if _temp:
        temp = _temp
        if debug:
            print(f"    Temperature: {temp}")
    elif _time:
        time = _time
        if debug:
            print(f"    Time: {time}")
    elif _state:
        state = _state
        if debug:
            print(f"    State: {state}")
    else:
        if debug:
            print(f"    Unrecognized: {display}")
            
    last_was_state = _state is not None
    
if temp is None and time is None and state is None:
    sys.stderr.write("ERROR: Failed to extract any of the temperature, time, or state.\n")
        
    with open(error_log_fn, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] ERROR: Failed to extract any of the temperature, time, or state\n")
        
    exit(1)

#
# 5. Publish the results to MQTT
#
# home/kiln/temperature is a temperature sensor in home-assistant
# that bridges with HomeKit

mqtt_broker = os.getenv('MQTT_BROKER')
mqtt_port = int(os.getenv('MQTT_PORT'))
mqtt_username = os.getenv('MQTT_USERNAME')
mqtt_password = os.getenv('MQTT_PASSWORD')

client = mqtt.Client(protocol=mqtt.MQTTv5)

# Set username and password if provided
if mqtt_username and mqtt_password:
    client.username_pw_set(mqtt_username, mqtt_password)

# Connect to MQTT broker
client.connect(mqtt_broker, mqtt_port, keepalive=60)

# Publish the temperature
if temp is not None:
    mqtt_topic = 'home/kiln/temperature'
    client.publish(mqtt_topic, temp)
    if debug:
        print(f"Published temperature: {temp} °F to topic: {mqtt_topic}")

# Publish the time
if time is not None:
    mqtt_topic = 'home/kiln/time'
    client.publish(mqtt_topic, time)
    if debug:
        print(f"Published time: {time} to topic: {mqtt_topic}")
    
# Publish the state
if state is not None:
    mqtt_topic = 'home/kiln/state'
    client.publish(mqtt_topic, state)
    if debug:
        print(f"Published state: {state} to topic: {mqtt_topic}")

# Disconnect
client.disconnect()

with open(run_log_fn, 'a') as f:
    f.write(f'[{datetime.now().isoformat()}] temp={temp}, time={time}, state={state}\n')

print(f"state={state}, temp={temp}, time={time}")

#
# 6. Publish to Firebase
#

FIREBASE_API_FILE = os.getenv('FIREBASE_API_FILE')

if FIREBASE_API_FILE is not None and state is not None and temp is not None and time is not None:
    import firebase_admin
    from firebase_admin import firestore, credentials
    import requests

    # fetch ambient temperature from Home Assistant
    ambient_temp = None
    
    HOMEASSISTANT_IP = os.getenv('HOMEASSISTANT_IP')
    HOMEASSISTANT_TARGET_ENTITY = os.getenv('HOMEASSISTANT_TARGET_ENTITY')
    HOMEASSISTANT_LONGLIVED_TOKEN = os.getenv('HOMEASSISTANT_LONGLIVED_TOKEN')
    
    url = f"http://{HOMEASSISTANT_IP}:8123/api/states/{HOMEASSISTANT_TARGET_ENTITY}"
    headers = {
        "Authorization": f"Bearer {HOMEASSISTANT_LONGLIVED_TOKEN}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        try:
            ambient_temp = float(data['state'])
        except ValueError:
            if debug:
                print(f"Failed to convert {data['state']} retrieved from home assistant to float")
            pass
        
    # publish to firebase
    doc = dict(
        run_time=time,
        state=state,
        temperature=temp,
        ambient_temperature=ambient_temp,
        time=datetime.now().isoformat(),
    )
    cred = credentials.Certificate(_join(thisdir, FIREBASE_API_FILE))
    firebase_admin.initialize_app(cred)

    db = firestore.client()
    db.collection('kiln_data').add(doc)

    if debug:
        print("Published to firebase", doc)
