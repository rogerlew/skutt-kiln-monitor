""""
    SecoDec - Simple, effective OCR for seven-segment displays

    Author: Scott Mudge - https://scottmudge.com

    Requires `opencv-python'

"""

import cv2
import sys
import os

import numpy
import numpy as np

debug = 1

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def preprocess_image(inp: np.ndarray) -> np.ndarray:
    return apply_brightness_contrast(inp, 0, 20)

"""
Segment Mask - Segments are index like so:

    ####          0 0
   #    #       5  8  1
   #    #       5  8  1
    ####   ==>    6 7
   #    #       4  9  2
   #    #       4  9  2
    ####          3 3

    The segment mask indicates which segments are active
    during each displayed number.
"""
segment_masks = {
        # 0  1  2  3  4  5  6  7  8  9
    "0": (1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
    "1": (0, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    "2": (1, 1, 0, 1, 1, 0, 1, 1, 0, 0),
    "3": (1, 1, 1, 1, 0, 0, 1, 1, 0, 0),
    "4": (0, 1, 1, 0, 0, 1, 1, 1, 0, 0),
    "5": (1, 0, 1, 1, 0, 1, 1, 1, 0, 0),
    "6": (1, 0, 1, 1, 1, 1, 1, 1, 0, 0),
    "7": (1, 1, 1, 0, 0, 1, 0, 0, 0, 0),
    "8": (1, 1, 1, 1, 1, 1, 1, 1, 0, 0),
    "9": (1, 1, 1, 1, 0, 1, 1, 1, 0, 0),
    "C": (1, 0, 0, 1, 1, 1, 0, 0, 0, 0),
    "P": (1, 1, 0, 0, 1, 1, 1, 1, 0, 0),
    "L": (0, 0, 0, 1, 1, 1, 0, 0, 0, 0),
    "T": (1, 0, 0, 0, 0, 0, 0, 0, 1, 1)
}

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

def otsu_like_threshold(background_pts, segment_pts):
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
        
        print(x2, var1, n1,  var2, n2, value)

        if value < min_value:
            min_value = value
            best_threshold = x2[0]-1 

    return best_threshold

def read_segments(image: np.ndarray, origin: tuple) -> list:
    x, y = origin
    
    background_pts = []
    for offset in background_offsets:
        x_offset, y_offset = offset
        x_offset += x
        y_offset += y
        value = image[y_offset, x_offset]
        background_pts.append(value)
    
    mu = np.mean(background_pts)
    sigma = np.std(background_pts)
    ci = 4 * sigma
    threshold = mu + ci
    
    if debug:
        print("Background Points", background_pts)
        print("Mean:", mu)
        print("Std Dev:", sigma)
        print("Threshold:", threshold)
        
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    segments = []
    segment_pts = []
    otus_pts = []
    for offset in segment_offsets:
        x_offset, y_offset = offset
        x_offset += x
        y_offset += y
        
        value = image[y_offset, x_offset]
        segment_pts.append(value)
        segments.append(int(value > threshold))
        
        otsu = otsu_thresh[y_offset, x_offset] 
        otus_pts.append(int(otsu / 255))
        
    if debug:
        print("Segment Points:", segment_pts)
        print("Otsu Points:", otus_pts)
        
        
    o2_threshold = otsu_like_threshold(background_pts, segment_pts)
    
    segments2 = []
    for value in segment_pts:
        segments2.append(int(value > o2_threshold))
    
    if debug:
        print("O2 Threshold:", o2_threshold)
        print("Segments2:", segments2)
        
    return segments2 #,segments

def read_char(image: np.ndarray, origin: tuple) -> str:
    segments = read_segments(image, origin)
    print(segments)
    
    for char, mask in segment_masks.items():
        if mask == tuple(segments):
            return char
        
    return "?"

def read_display(image: np.ndarray) -> str:
    output = ""
    for i, char_origin in enumerate(char_origins):
        if debug:
            print("Reading char", i)
        output += read_char(image, char_origin)
        
    return output


if __name__ == "__main__":
    fn = "test_cplt.jpg"
    
    image = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    #image = preprocess_image(image)
    
    annotated = image.copy()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    for char_origin in char_origins:
        print(char_origin)
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
        
        for offset in segment_offsets:
            x, y = offset
            x += ul_x
            y += ul_y
            cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
            
        for background_offset in background_offsets:
            x, y = background_offset
            x += ul_x
            y += ul_y
            cv2.circle(annotated, (x, y), 5, (255, 0, 0), -1)
        
    cv2.imwrite("secodec_00_annotated.jpg", annotated)
    
    image = preprocess_image(image)
    
    if debug:
        cv2.imwrite("secodec_01_preprocessed.jpg", image)
    
    display = read_display(image)
    print(display)