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

import paho.mqtt.client as mqtt

# This purposefully uses a script approach because it is easier to debug and not get lost.
# With these sort of CV scripts, there is alot of trial and error and complex program flows
# end up causing problems if it isn't doing what you actually think it is doing.
#
# I also knew at the outset I would be deploying it as crontab on Linux. So the scripted
# approach is more robust for that purpose. 

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

# The Skutt kiln cycles through state, temperature, and time, we collect 7 frames
# so that we can be sure to observe state -> temperature event
n_frames = 7  

if capture_method == "frame_interval":
    # Frame interval (e.g., 1 frame every 2 seconds)
    frame_rate = 0.35  # Capture 0.5 fps (1 frame every 2 seconds)

    # FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-rtsp_transport", "tcp",    # Use TCP for stability
        "-i", rtsp_url,              # Input RTSP URL
        "-vf", f"fps={frame_rate}",  # Set the capture frame rate
        "-frames:v", f"{n_frames}",            # Capture 5 frames
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
        "-frames:v", f"{n_frames}",          # Capture 5 frames
        output_pattern,            # Output file pattern
        "-y"                       # Overwrite existing files
    ]
else:
    print("ERROR: Invalid capture method")
    exit(1)


print(' '.join(ffmpeg_command))

#
# 2.1 Remove existing frames
output_frames = glob.glob("frame_*.jpg")
for output_frame in output_frames:
    os.remove(output_frame)

#
# 2.2 Capture the RTSP stream
try:
    # Run FFmpeg command
    subprocess.run(ffmpeg_command, check=True, timeout=4*n_frames)
except subprocess.CalledProcessError as e:
    print(f"FFmpeg failed: {e}")

# Check if frames were captured
output_frames = glob.glob("frame_*.jpg")
assert len(output_frames) > 0, "ERROR: No frames captured"

if debug:
    print(f"Captured {len(output_frames)} frames\n\n")

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
#
# It seems to be functional, but could be improved in the future

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
                
    assert H_box is not None, "ERROR: Cannot determine homography"
        
        
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
    rectified_box = cv2.warpPerspective(weighted_gray, H_box, (output_width, output_height))

    if debug > 1:
        cv2.imwrite(f"frame_{k+1:04d},03_rectified.jpg", rectified_box)

    if H_fine is None:
        if debug:
            print("  Detecting features for fine homography...")
            
        template = cv2.imread("../templates/template.jpg", cv2.IMREAD_GRAYSCALE)
        
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
            print("  Homography identification successful")

        # Serialize homography to json
        homography = {
            "H": H_fine.tolist(),
            "output_width": output_width,
            "output_height": output_height
        }
        with open("H_fine.json", "w") as f:
            json.dump(homography, f)

    #
    # 3.2.2 Homography Fallback
    if H_fine is None:
        # this occurs in low-light conditions, so the camera could be setup
        # with good lighting and then after the lights go off the camera
        # can still retrieve the image from the last successful homography
        if debug:
            print("Feature detection homography failed")
        if  _exists("H_fine.json"):
            if debug:
                print("Using last successful homography")
            with open("H_fine.json", "r") as f:
                homography = json.load(f)
                H_fine = np.array(homography["H"], dtype=np.float32)
                output_width = homography["output_width"]
                output_height = homography["output_height"]
                
    assert H_fine is not None, "ERROR: Cannot determine homography"

    h_t, w_t = template.shape[:2]
    corrected_box = cv2.warpPerspective(rectified_box, H_fine, (w_t, h_t))

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
        
#
# 4. Read the processed frames
#
# this is loosely based on the approach by https://github.com/scottmudge/SegoDec/


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

segment_masks = {
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

# these specfies the origin of the 4 characters in the display in pixels coordinates
# the origin is the top-left corner of the character
char_origins = [
    (52-8, 58-8), # (131, 202)
    (178-8, 58-8),
    (304-5, 58-8),
    (434-8, 58-8)
]

# these roughly define the width and height of the character in pixels
# but the character actually extends beyond these points a bit.
char_width = 135 - 52
char_height = 202 - 58

# These define the segment sampling points.
# The sampling points are used to determine if the segment is active.
segment_offsets = [
    (100-52,  68-58),   # 0
    (262-178, 98-58),   # 1
    (256-178, 158-58),  # 2
    (91-52,   192-58),  # 3
    (59-52,   158-58),  # 4
    (63-52,   98-58),   # 5
    (207-178, 129-58),  # 6
    (241-178, 129-58),  # 7
    (480-434, 98-58),   # 8
    (474-434, 158-58),  # 9
    (460-429, 84-58),   # 10
    (450-429, 168-58),  # 11
    (499-429, 84-58),   # 12
    (490-434, 168-58),  # 13
    (260-170, 181-58),  # 14
]

# normalize to character width and height
# tried to normalize to make it less size dependent
segment_offsets = [(x / char_width, y / char_height) for x, y in segment_offsets]

# These define the background sampling points around each segment.
# The idea is that we know these are background and can use there values
# to constrast active segments.
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

background_offsets = [(x / char_width, y / char_height) for x, y in background_offsets]

## Some Helper functions to avoid heavy nesting

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
    global debug, segment_offsets, background_offsets, char_width, char_height
    
    if debug:
        print("read_segments", origin)
        
    assert method in ['confidence_interval', 'point_otsu', 'otsu', 'ensemble'], "ERROR: Invalid method"
    
    origin_x, origin_y = origin
    
    background_pts = []
    for x_offset, y_offset in background_offsets:
        _x = int(origin_x + x_offset * char_width)
        _y = int(origin_y + y_offset * char_height)
        background_pts.append(image[_y, _x])
    
    segment_pts = []
    for x_offset, y_offset in segment_offsets:
        _x = int(origin_x + x_offset * char_width)
        _y = int(origin_y + y_offset * char_height)
        segment_pts.append(image[_y, _x])
        
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
            _x = int(origin_x + x_offset * char_width)
            _y = int(origin_y + y_offset * char_height)
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
    global segment_masks
    
    segments = read_segments(image, origin)
    
    if (segments[8] == 1 and segments[9] == 1):
        if debug:
            print("  WARNING: setting segments 10, 11, 12, 13 to 0 due to 8 and 9 being active")
            
        segments[10] = 0
        segments[11] = 0
        segments[12] = 0
        segments[13] = 0
    
    ret = ''
    
    n = len(segments[:-1])
    match_counts = {}
    for char, mask in segment_masks.items():
        # need to ignore the period segment for now or it messes up the matching counts
        if char == '.':
            continue
        match_counts[char] = n - sum(m == s for m, s in zip(mask[:-1], segments[:-1]))
            
    # Find the best match and append to ret is within the error tolerance
    best_match = min(match_counts, key=match_counts.get)
    
    if debug > 4:
        print("    Match Counts:", match_counts)
        print("    Best Match:", best_match)
    
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
            
            for x, y in segment_offsets:
                x = int(x * char_width + ul_x)
                y = int(y * char_height + ul_y)
                cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
                
            for x, y in background_offsets:
                x = int(x * char_width + ul_x)
                y = int(y * char_height + ul_y)
                cv2.circle(annotated, (x, y), 5, (255, 0, 0), -1)
        
        cv2.imwrite(f"frame_{k+1:04d},06_char_segmentation.jpg", annotated)
    
    display = read_display(image)
    
    if debug:
        print(f"  Display: {display}\n")
        
    displays.append(display)
    
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
    
if debug:
    if not temp or not time or not state:
        print("ERROR: Failed to extract any of the temperature, time, or state.")
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
    print(f"Published temperature: {temp} °F to topic: {mqtt_topic}")

# Publish the time
if time is not None:
    mqtt_topic = 'home/kiln/time'
    client.publish(mqtt_topic, time)
    print(f"Published time: {time} to topic: {mqtt_topic}")
    
# Publish the state
if state is not None:
    mqtt_topic = 'home/kiln/state'
    client.publish(mqtt_topic, state)
    print(f"Published state: {state} to topic: {mqtt_topic}")

# Disconnect
client.disconnect()
