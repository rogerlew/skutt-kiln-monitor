# Skutt Kiln LCD Computer Vision Reader

A Python-based Computer Vision (CV) tool designed to read and interpret the state of a Skutt kiln’s segmented LCD display using a low-cost IP or RTSP camera feed. This script:

1. Captures frames from a Tapo or similar RTSP camera.
2. Uses an ArUco marker to establish a homography and rectify the image.
3. Applies feature-based matching to further refine the image alignment.
4. Identifies active segments of the kiln’s LCD display.
5. Derives the kiln’s current temperature, firing state, and the displayed time.
6. Publishes these values to MQTT topi

## Features

- Captures images from an RTSP camera feed.
- Uses ArUco markers for homography-based image rectification.
- Employs template matching and feature detection for fine-tuned alignment.
- Reads segmented digital displays to extract temperature, time, and kiln state.
- Publishes extracted data to an MQTT broker for integration into home automation systems.
- Supports flexible debugging options with image outputs at various processing stages.

**License:** [BSD-3-Clause](./LICENSE)

---

## Features

- **Robust Alignment:**  
  Utilizes ArUco fiducials and feature-based matching (AKAZE) to correct for perspective distortions and lens issues.
- **Segmented LCD Detection:**  
  Analyzes which LCD segments are active, applying logical rules and various thresholding methods (confidence intervals, Otsu-like approaches) to handle challenging layouts.
- **MQTT Integration:**  
  Publishes temperature, time, and firing state readings to MQTT topics, simplifying integration with home automation frameworks.

---

## Setup & Requirements

1. **Camera Setup:**  
   Position a webcam or RTSP-capable IP camera close to the kiln’s LCD and the attached ArUco marker (ID=8). A stable, front-facing view ensures accurate homography and segment detection.

2. **Dependencies:**
   - Python 3.x
   - OpenCV (with contrib modules for ArUco):  
     ```bash
     pip install opencv-python opencv-contrib-python
     ```
   - MQTT client:  
     ```bash
     pip install paho-mqtt
     ```
   - Environment variables support:  
     ```bash
     pip install python-dotenv
     ```
   - FFmpeg (installed and on PATH for RTSP frame capture).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Configure the `.env` file with your credentials and settings:
   ```dotenv
   TAPO_USERNAME=<username>
   PASSWORD=<password>
   RTSP_IP=<camera-ip>
   MQTT_BROKER=<broker-ip>
   MQTT_PORT=<broker-port>
   MQTT_USERNAME=<mqtt-username>
   MQTT_PASSWORD=<mqtt-password>
   ```
   
## Usage

Run the script from the command line:
```bash
python skutt_read.py
```

## Debugging & Customization

**The script supports three levels of debugging:**

- **Level 0**: No debug information.
- **Level 1**: Console output.
- **Level 2**: Image outputs at key processing stages (e.g., grayscale conversion, rectification, character segmentation).

**Fine tune `bounding box` for LCD Segment ROI and `char_origins`:**

```
cropped = corrected_box[653:653+250, 664:664+565]
```

**Update Template File**

The template file may need to be updated for the fine homography adjustment.


## Physical Setup


## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE.md) file for details.

---

## Visual Documentation

### Physical Setup

*(Placeholder for an image showing the camera and kiln setup)*

### Computer Vision Processing

#### Grayscale Conversion

*(Placeholder for an image of a grayscale frame)*

#### ArUco Marker Detection

*(Placeholder for an image showing marker detection)*

#### Image Rectification

*(Placeholder for an image of the rectified ROI)*

#### Character Segmentation

*(Placeholder for an annotated image showing character segmentation)*

#### Final Data Extraction

*(Placeholder for a sample of extracted data)*
