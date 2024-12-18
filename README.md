# Skutt Kiln LCD Computer Vision Reader

A Python-based Computer Vision (CV) tool designed to read and interpret the state of a Skutt kiln’s segmented LCD display using a low-cost IP or RTSP camera feed. This script:

1. Captures frames from a Tapo or similar RTSP camera.
2. Uses an ArUco marker to establish a homography and rectify the image.
3. Applies feature-based matching to further refine the image alignment.
4. Identifies active segments of the kiln’s LCD display.
5. Derives the kiln’s current temperature, firing state, and the displayed time.
6. Publishes these values to MQTT topic for home-assistant/HomeKit integration

## Features

- Captures images from an RTSP camera feed.
- Uses ArUco markers for homography-based image rectification.
- Employs template matching and feature detection for fine-tuned alignment.
- Reads segmented digital displays to extract temperature, time, and kiln state.
- Publishes extracted data to an MQTT broker for integration into home automation systems.
- Supports flexible debugging options with image outputs at various processing stages.

**License:** [BSD-3-Clause](LICENSE.md)

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

### Computer Vision Processing

#### Grayscale Conversion

![frame_0001,01_gray](https://github.com/user-attachments/assets/e19e9a2e-cd9d-4654-82c9-0404d8885a68)

#### ArUco Marker Detection

![frame_0001,02_annotated_frame](https://github.com/user-attachments/assets/06c96c76-9648-44da-bf26-6bef0b4669b4)

#### Image Rectification

![frame_0001,03_rectified](https://github.com/user-attachments/assets/22a97cc4-dd2d-4ab3-acc7-933256d34d82)

#### Template Feature Mapping

![frame_0001,04_feature_matching](https://github.com/user-attachments/assets/8e8e4834-da1a-4c82-bbad-51639b170d6f)

#### Image Rectification (2-pass)

![frame_0001,05_corrected](https://github.com/user-attachments/assets/c63f277a-59f4-4563-b65e-87638be67f06)

#### Character Segmentation

![frame_0001,06_char_segmentation](https://github.com/user-attachments/assets/d8d98ab5-f605-41c2-8a48-c309d70ac9a3)

#### LCD Display ROI for reading

![frame_0001,06_cropped](https://github.com/user-attachments/assets/f0827252-2566-4653-a9fb-c046faee54b2)

## Example Output

```
...
Captured 7 frames


Processing frame 0...
  Detecting markers...
Marker not found
Using last successful homography
  Rectifying image...
  Grayscale conversion,  Raw Min Value: 0.0, Raw Max Value: 225.216
  Detecting features for fine homography...
  Found 133 good matches
  Homography identification successful
  Corrected image saved as frame_0001,05_corrected.jpg

Processing frame 1...
  Rectifying image...
  Grayscale conversion,  Raw Min Value: 0.0, Raw Max Value: 225.216
  Corrected image saved as frame_0002,05_corrected.jpg

Processing frame 2...
  Rectifying image...
  Grayscale conversion,  Raw Min Value: 0.0, Raw Max Value: 225.216
  Corrected image saved as frame_0003,05_corrected.jpg

Processing frame 3...
  Rectifying image...
  Grayscale conversion,  Raw Min Value: 0.0, Raw Max Value: 225.216
  Corrected image saved as frame_0004,05_corrected.jpg

Processing frame 4...
  Rectifying image...
  Grayscale conversion,  Raw Min Value: 0.0, Raw Max Value: 225.216
  Corrected image saved as frame_0005,05_corrected.jpg

Processing frame 5...
  Rectifying image...
  Grayscale conversion,  Raw Min Value: 0.0, Raw Max Value: 225.216
  Corrected image saved as frame_0006,05_corrected.jpg

Processing frame 6...
  Rectifying image...
  Grayscale conversion,  Raw Min Value: 0.0, Raw Max Value: 225.216
  Corrected image saved as frame_0007,05_corrected.jpg

Reading processed frames...
  Reading frame 1...
Reading char 1 @ (44, 50)
read_segments (44, 50)
  Background Points [78, 118, 86, 78, 109, 71, 110, 143, 99, 95]
  Segment Points: [253, 101, 88, 253, 250, 251, 111, 87, 106, 99, 151, 154, 126, 105, 86]
  Otsu Point Segmentation
    Threshold: 249
    Segments: [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 2 @ (170, 50)
read_segments (170, 50)
  Background Points [83, 125, 88, 81, 73, 79, 114, 135, 146, 99]
  Segment Points: [253, 251, 112, 95, 253, 254, 254, 253, 135, 135, 165, 159, 236, 98, 87]
  Otsu Point Segmentation
    Threshold: 235
    Segments: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
Reading char 3 @ (299, 50)
read_segments (299, 50)
  Background Points [87, 79, 78, 88, 105, 73, 117, 129, 74, 76]
  Segment Points: [89, 80, 87, 254, 254, 254, 123, 82, 99, 109, 128, 166, 79, 103, 83]
  Otsu Point Segmentation
    Threshold: 253
    Segments: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 4 @ (426, 50)
read_segments (426, 50)
  Background Points [78, 138, 82, 71, 69, 62, 82, 78, 80, 69]
  Segment Points: [253, 97, 92, 124, 98, 103, 150, 128, 254, 252, 151, 146, 154, 159, 76]
  Otsu Point Segmentation
    Threshold: 251
    Segments: [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  WARNING: setting segments 10, 11, 12, 13 to 0 due to 8 and 9 being active
  Display: CPLT

  Reading frame 2...
Reading char 1 @ (44, 50)
read_segments (44, 50)
  Background Points [61, 59, 59, 53, 53, 57, 61, 57, 64, 60]
  Segment Points: [63, 65, 59, 57, 58, 61, 60, 62, 62, 59, 62, 58, 64, 60, 59]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 2 @ (170, 50)
read_segments (170, 50)
  Background Points [60, 67, 71, 64, 71, 78, 61, 60, 86, 96]
  Segment Points: [69, 75, 73, 71, 64, 65, 64, 67, 66, 68, 65, 65, 66, 71, 83]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 3 @ (299, 50)
read_segments (299, 50)
  Background Points [80, 129, 90, 89, 115, 88, 107, 116, 107, 128]
  Segment Points: [252, 119, 251, 254, 254, 254, 253, 254, 131, 144, 159, 160, 130, 174, 135]
  Otsu Point Segmentation
    Threshold: 250
    Segments: [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
Reading char 4 @ (426, 50)
read_segments (426, 50)
  Background Points [87, 129, 87, 97, 109, 70, 126, 143, 142, 101]
  Segment Points: [254, 254, 254, 254, 253, 253, 254, 254, 158, 148, 180, 174, 242, 180, 138]
  Otsu Point Segmentation
    Threshold: 241
    Segments: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  Display:   68

  Reading frame 3...
Reading char 1 @ (44, 50)
read_segments (44, 50)
  Background Points [56, 63, 61, 50, 55, 56, 59, 57, 64, 58]
  Segment Points: [64, 65, 58, 57, 60, 62, 62, 62, 64, 60, 60, 59, 64, 59, 59]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 2 @ (170, 50)
read_segments (170, 50)
  Background Points [59, 65, 69, 65, 69, 77, 64, 58, 90, 95]
  Segment Points: [66, 73, 76, 71, 64, 63, 64, 67, 63, 66, 64, 66, 67, 71, 81]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 3 @ (299, 50)
read_segments (299, 50)
  Background Points [85, 129, 86, 86, 110, 93, 105, 112, 108, 138]
  Segment Points: [252, 126, 253, 255, 254, 254, 252, 253, 142, 152, 153, 168, 126, 170, 146]
  Otsu Point Segmentation
    Threshold: 251
    Segments: [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
Reading char 4 @ (426, 50)
read_segments (426, 50)
  Background Points [85, 130, 85, 94, 111, 72, 119, 142, 131, 107]
  Segment Points: [254, 254, 253, 254, 254, 252, 253, 253, 156, 152, 166, 185, 241, 164, 147]
  Otsu Point Segmentation
    Threshold: 240
    Segments: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  Display:   68

  Reading frame 4...
Reading char 1 @ (44, 50)
read_segments (44, 50)
  Background Points [59, 62, 65, 55, 58, 67, 58, 56, 72, 92]
  Segment Points: [66, 73, 71, 62, 61, 63, 64, 65, 66, 63, 65, 61, 68, 62, 76]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 2 @ (170, 50)
read_segments (170, 50)
  Background Points [69, 125, 98, 82, 118, 90, 80, 126, 146, 100]
  Segment Points: [253, 252, 134, 253, 252, 114, 253, 254, 135, 138, 135, 170, 229, 131, 253]
  Otsu Point Segmentation
    Threshold: 228
    Segments: [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]
Reading char 3 @ (299, 50)
read_segments (299, 50)
  Background Points [88, 89, 91, 82, 74, 88, 118, 101, 148, 132]
  Segment Points: [113, 253, 253, 93, 104, 253, 254, 254, 138, 129, 135, 101, 230, 156, 131]
  Otsu Point Segmentation
    Threshold: 229
    Segments: [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
Reading char 4 @ (426, 50)
read_segments (426, 50)
  Background Points [91, 124, 93, 94, 105, 74, 134, 138, 135, 106]
  Segment Points: [253, 253, 254, 253, 253, 253, 251, 253, 151, 155, 174, 172, 239, 177, 148]
  Otsu Point Segmentation
    Threshold: 238
    Segments: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  Display:  2.48

  Reading frame 5...
Reading char 1 @ (44, 50)
read_segments (44, 50)
  Background Points [76, 112, 85, 81, 112, 74, 106, 148, 100, 102]
  Segment Points: [253, 98, 90, 253, 252, 254, 114, 84, 112, 104, 141, 143, 118, 112, 82]
  Otsu Point Segmentation
    Threshold: 251
    Segments: [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 2 @ (170, 50)
read_segments (170, 50)
  Background Points [84, 125, 91, 79, 75, 80, 116, 121, 132, 104]
  Segment Points: [253, 250, 106, 92, 253, 253, 251, 251, 141, 136, 160, 160, 229, 103, 88]
  Otsu Point Segmentation
    Threshold: 228
    Segments: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
Reading char 3 @ (299, 50)
read_segments (299, 50)
  Background Points [84, 84, 75, 85, 100, 73, 111, 121, 75, 79]
  Segment Points: [89, 83, 84, 254, 252, 254, 124, 83, 96, 108, 131, 165, 81, 106, 84]
  Otsu Point Segmentation
    Threshold: 251
    Segments: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 4 @ (426, 50)
read_segments (426, 50)
  Background Points [81, 143, 85, 71, 72, 61, 80, 78, 83, 68]
  Segment Points: [253, 99, 90, 128, 102, 104, 144, 129, 253, 254, 170, 148, 157, 163, 75]
  Otsu Point Segmentation
    Threshold: 252
    Segments: [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  WARNING: setting segments 10, 11, 12, 13 to 0 due to 8 and 9 being active
  Display: CPLT

  Reading frame 6...
Reading char 1 @ (44, 50)
read_segments (44, 50)
  Background Points [74, 124, 87, 89, 120, 73, 103, 140, 100, 96]
  Segment Points: [255, 102, 91, 253, 254, 253, 118, 88, 115, 103, 152, 149, 128, 107, 91]
  Otsu Point Segmentation
    Threshold: 252
    Segments: [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 2 @ (170, 50)
read_segments (170, 50)
  Background Points [87, 121, 93, 82, 77, 75, 112, 125, 133, 102]
  Segment Points: [254, 251, 117, 97, 252, 254, 254, 254, 144, 120, 168, 157, 234, 102, 89]
  Otsu Point Segmentation
    Threshold: 233
    Segments: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
Reading char 3 @ (299, 50)
read_segments (299, 50)
  Background Points [88, 80, 75, 88, 103, 72, 105, 120, 78, 78]
  Segment Points: [91, 79, 86, 254, 253, 253, 113, 81, 93, 110, 128, 158, 78, 103, 80]
  Otsu Point Segmentation
    Threshold: 252
    Segments: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 4 @ (426, 50)
read_segments (426, 50)
  Background Points [75, 119, 85, 72, 71, 62, 81, 79, 81, 68]
  Segment Points: [254, 99, 93, 113, 99, 101, 147, 124, 253, 254, 156, 143, 150, 156, 77]
  Otsu Point Segmentation
    Threshold: 252
    Segments: [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  WARNING: setting segments 10, 11, 12, 13 to 0 due to 8 and 9 being active
  Display: CPLT

  Reading frame 7...
Reading char 1 @ (44, 50)
read_segments (44, 50)
  Background Points [59, 59, 63, 52, 54, 57, 57, 56, 63, 58]
  Segment Points: [64, 64, 59, 57, 61, 61, 59, 63, 62, 59, 59, 59, 64, 58, 58]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 2 @ (170, 50)
read_segments (170, 50)
  Background Points [63, 64, 71, 60, 71, 76, 63, 57, 90, 94]
  Segment Points: [70, 78, 76, 72, 63, 63, 64, 71, 64, 65, 66, 64, 69, 69, 83]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Reading char 3 @ (299, 50)
read_segments (299, 50)
  Background Points [87, 141, 85, 85, 106, 88, 103, 118, 110, 132]
  Segment Points: [252, 127, 252, 252, 254, 253, 253, 254, 136, 141, 162, 170, 125, 177, 144]
  Otsu Point Segmentation
    Threshold: 251
    Segments: [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
Reading char 4 @ (426, 50)
read_segments (426, 50)
  Background Points [87, 130, 87, 93, 109, 69, 128, 136, 136, 106]
  Segment Points: [254, 253, 253, 252, 253, 252, 252, 253, 149, 150, 172, 178, 240, 168, 139]
  Otsu Point Segmentation
    Threshold: 239
    Segments: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  Display:   68

Frame Semantics
  Frame 1: CPLT
    State: Complete
  Frame 2:   68
    Temperature: 68.0
  Frame 3:   68
    Unrecognized:   68
  Frame 4:  2.48
    Time: 02:48
  Frame 5: CPLT
    State: Complete
  Frame 6: CPLT
    State: Complete
  Frame 7:   68
    Temperature: 68.0

Published temperature: 68.0 °F to topic: home/kiln/temperature
Published time: 02:48 to topic: home/kiln/time
Published state: Complete to topic: home/kiln/state
```


