# Skutt Kiln LCD Computer Vision Reader

A Python-based Computer Vision (CV) tool designed to monitor the state of a Skutt kiln’s segmented LCD display using a low-cost IP or RTSP camera feed. This script:

1. Captures frames from a Tapo or similar RTSP camera.
2. Uses an ArUco marker to establish a homography and rectify the image.
3. Applies feature-based matching to further refine the image alignment.
4. Composites images and uses contours to revise character origins.
5. Identifies active segments of the kiln’s LCD display.
6. Derives the kiln’s current temperature, firing state, and the displayed time.
7. Publishes these values to MQTT topic for home-assistant/HomeKit integration

![image](https://github.com/user-attachments/assets/577bdf7b-e2ec-4986-b987-a152cf80c4b7)

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
- **Inexpensive:**
  Utilizes sub-$20 commodity RTSP web camera
  
Tapo (TP-Link) C 110
![image](https://github.com/user-attachments/assets/f661f353-2514-4d06-b96e-3c75f2611396)

---

## Setup & Requirements

1. **Camera Setup:**  
   Position a webcam or RTSP-capable IP camera close to the kiln’s LCD and the attached ArUco marker (ID=8) 2"x2". A stable, front-facing view ensures accurate homography and segment detection.

![4x4_1000-8](https://github.com/user-attachments/assets/3417e96e-64f5-4a3f-bffa-3d01a3d9f6ed)

2. **Dependencies:**
   - Tested with Python 3.12 conda
   ```bash
   conda create --name skutt-monitor python=3.12
   conda activate skutt-monitor
   conda install -c conda-forge opencv
   conda install -c conda-forge ffmpeg
   conda install -c conda-forge paho-mqtt
   conda install -c conda-forge python-dotenv
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rogerlew/skutt_kiln_cv
   cd skutt_kiln_cv
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

3. Optionally create ramdisk at /ramdisk to write image files. script sets this to the
   working directory if it exists. Otherwise it creates a working dir relative to current
   directory.
   
## Usage

Run the script from the command line:
```bash
conda run -n skutt-monitor python /workdir/skutt-kiln-monitor/skutt_read.py --no_debug
```
or to suppress debug

```bash
conda run -n skutt-monitor python /workdir/skutt-kiln-monitor/skutt_read.py --no_debug
```

## Deployment

- Create `/var/log/skutt-monitor` folder with permissions to write
- add crontab
```
*/10 * * * * conda run -n skutt-monitor python /workdir/skutt-kiln-monitor/skutt_read.py --no_debug
```

## Debugging & Customization

**The script supports three levels of debugging:**

- **Level 0**: No debug information.
- **Level 1**: Console output.
- **Level 2**: Image outputs at key processing stages (e.g., grayscale conversion, rectification, character segmentation).

**Fine tune `bounding box` for LCD Segment ROI and `char_origins`:**

```python
cropped = corrected_box[653:653+250, 664:664+565]
```

**Update Template File**

The template file (`templates/templates.jpg`) may need to be updated for the fine homography adjustment. The template file should have the LCD Display as rectilinear as possible.
Web cameras with large fields of view have a great deal of lens distortion. To create the template image the camera was positioned 16" from the LCD 
screen such that the lens was centered vertically and horizontally with the screen. The image was captured and cropped to exclude background elements.

**Home Assistant Integration**

<img src="https://github.com/user-attachments/assets/7e792e81-092a-4f7f-98fe-f208f22815fb" alt="Home Assistant Logo" width="180"/>

```yaml
sensor:
  - platform: mqtt
    name: "Kiln Temperature"
    state_topic: "home/kiln/temperature"
    unit_of_measurement: "°F"
    value_template: "{{ value }}"
    device_class: "temperature"
    unique_id: "kiln_temperature_sensor"
```

<img src="https://github.com/user-attachments/assets/c141efe5-22fc-446e-a438-350312130355" alt="HomeKit Logo" width="200"/>

For HomeKit bridge add to `sensor.kiln_temperature` to `include_entities` list


## Physical Setup

<img src="https://github.com/user-attachments/assets/2787d67a-0d74-4512-8686-47f12ead77c9" alt="image" width="300"/>


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

#### Precise Contour Based Character Detection

The frames are composited using lighten blend.

![roi_composite_frame](https://github.com/user-attachments/assets/acbe7fb0-124e-4c9b-9f6f-5d3fbaf51a8a)

Then contours are identified and the intersecting bounding boxes are used to revise the character origins

![roi_bounding_boxes](https://github.com/user-attachments/assets/ba5b7f45-7122-486b-97ba-38f517b8e39f)

#### Character Segmentation

Original point based segmentation

![frame_0001,06_char_segmentation](https://github.com/user-attachments/assets/d8d98ab5-f605-41c2-8a48-c309d70ac9a3)

Revised mask based segmentation

![frame_0001,06_char_segmentation](https://github.com/user-attachments/assets/9f223635-96db-459d-a93d-8bfe18d10b47)

This is the template file used to define the segment masks

![segment_mask](https://github.com/user-attachments/assets/525aa7dd-4c15-4274-8ff3-6155d784ab60)

#### LCD Display ROI for reading

![frame_0001,06_cropped](https://github.com/user-attachments/assets/f0827252-2566-4653-a9fb-c046faee54b2)

## Example Output

```bash
...
Captured 7 frames


Processing frame 0...
  Detecting markers...
  Homography identification successful
  Rectifying image...
  Grayscale conversion,  Raw Min Value: 0.0, Raw Max Value: 225.216
  Detecting features for fine homography...
  Found 219 good matches
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

Revising character origin for character 1 at (44, 50) to (35, 47)
Revising character origin for character 2 at (170, 50) to (165, 47)
Revising character origin for character 3 at (299, 50) to (295, 47)
Revising character origin for character 4 at (426, 50) to (424, 47)
Reading processed frames...
  Reading frame 1...
Reading char 1 @ (35, 47)
read_segments (35, 47)
  Background Points [46, 52, 62, 61, 60, 49, 55, 57, 56, 54, 54, 53, 49, 64]
  Segment Points: [63, 68, 62, 64, 65, 67, 66, 64, 67, 66, 61, 64, 65, 64, 60]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match:
    Match Error: 0
Reading char 2 @ (165, 47)
read_segments (165, 47)
  Background Points [57, 57, 59, 60, 55, 51, 60, 60, 59, 54, 57, 58, 46, 60]
  Segment Points: [55, 58, 58, 61, 60, 61, 60, 56, 58, 60, 61, 60, 54, 62, 54]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match:
    Match Error: 0
Reading char 3 @ (295, 47)
read_segments (295, 47)
  Background Points [50, 51, 41, 47, 52, 54, 54, 51, 54, 54, 54, 51, 61, 54]
  Segment Points: [156, 154, 161, 54, 58, 58, 59, 58, 60, 59, 56, 59, 58, 54, 51]
  Otsu Point Segmentation
    Threshold: 153
    Segments: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match: 7
    Match Error: 0
Reading char 4 @ (424, 47)
read_segments (424, 47)
  Background Points [39, 52, 53, 36, 50, 39, 39, 45, 46, 50, 54, 50, 51, 46]
  Segment Points: [49, 148, 154, 55, 59, 52, 53, 52, 48, 55, 49, 53, 51, 52, 50]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match: 1
    Match Error: 0
  Display:   71

  Reading frame 2...
Reading char 1 @ (35, 47)
read_segments (35, 47)
  Background Points [60, 53, 56, 44, 56, 54, 58, 50, 56, 59, 46, 55, 58, 62]
  Segment Points: [67, 67, 66, 65, 67, 68, 67, 67, 69, 65, 66, 65, 65, 61, 61]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match:
    Match Error: 0
Reading char 2 @ (165, 47)
read_segments (165, 47)
  Background Points [55, 60, 42, 61, 54, 60, 53, 56, 50, 57, 54, 58, 58, 60]
  Segment Points: [158, 152, 60, 148, 163, 65, 179, 169, 60, 61, 64, 66, 59, 64, 156]
  Otsu Point Segmentation
    Threshold: 147
    Segments: [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    Best Match: 2
    Match Error: 0
Reading char 3 @ (295, 47)
read_segments (295, 47)
  Background Points [51, 59, 52, 49, 54, 48, 56, 55, 54, 56, 42, 61, 54, 55]
  Segment Points: [53, 153, 156, 56, 58, 157, 179, 176, 58, 60, 58, 60, 58, 55, 56]
  Otsu Point Segmentation
    Threshold: 152
    Segments: [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    Best Match: 4
    Match Error: 0
Reading char 4 @ (424, 47)
read_segments (424, 47)
  Background Points [52, 39, 36, 44, 49, 43, 52, 40, 44, 51, 50, 48, 41, 51]
  Segment Points: [146, 149, 151, 145, 152, 155, 171, 162, 51, 58, 52, 59, 51, 56, 52]
  Otsu Point Segmentation
    Threshold: 144
    Segments: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    Best Match: 8
    Match Error: 0
  Display:  2.48

  Reading frame 3...
Reading char 1 @ (35, 47)
read_segments (35, 47)
  Background Points [50, 57, 58, 57, 62, 57, 57, 60, 57, 63, 57, 56, 58, 61]
  Segment Points: [64, 68, 64, 64, 69, 67, 66, 68, 67, 66, 64, 65, 67, 62, 60]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match:
    Match Error: 0
Reading char 2 @ (165, 47)
read_segments (165, 47)
  Background Points [38, 58, 40, 61, 37, 52, 57, 62, 40, 59, 57, 55, 58, 58]
  Segment Points: [156, 152, 59, 148, 163, 65, 175, 167, 60, 62, 64, 65, 61, 65, 157]
  Otsu Point Segmentation
    Threshold: 147
    Segments: [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    Best Match: 2
    Match Error: 0
Reading char 3 @ (295, 47)
read_segments (295, 47)
  Background Points [52, 47, 50, 51, 53, 53, 54, 51, 58, 57, 47, 51, 50, 56]
  Segment Points: [55, 151, 154, 55, 61, 154, 182, 172, 59, 61, 56, 58, 54, 57, 52]
  Otsu Point Segmentation
    Threshold: 150
    Segments: [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    Best Match: 4
    Match Error: 0
Reading char 4 @ (424, 47)
read_segments (424, 47)
  Background Points [51, 59, 48, 47, 36, 38, 41, 51, 49, 37, 48, 49, 50, 50]
  Segment Points: [148, 149, 149, 146, 156, 156, 173, 163, 50, 60, 53, 58, 51, 57, 50]
  Otsu Point Segmentation
    Threshold: 145
    Segments: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    Best Match: 8
    Match Error: 0
  Display:  2.48

  Reading frame 4...
Reading char 1 @ (35, 47)
read_segments (35, 47)
  Background Points [49, 62, 59, 51, 52, 61, 54, 56, 59, 56, 50, 55, 60, 58]
  Segment Points: [154, 64, 63, 147, 154, 159, 66, 66, 67, 65, 65, 68, 67, 62, 58]
  Otsu Point Segmentation
    Threshold: 146
    Segments: [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match: C
    Match Error: 0
Reading char 2 @ (165, 47)
read_segments (165, 47)
  Background Points [60, 43, 44, 52, 61, 61, 62, 59, 37, 56, 58, 64, 59, 57]
  Segment Points: [156, 152, 61, 58, 162, 162, 178, 174, 61, 63, 65, 66, 61, 67, 50]
  Otsu Point Segmentation
    Threshold: 151
    Segments: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    Best Match: P
    Match Error: 0
Reading char 3 @ (295, 47)
read_segments (295, 47)
  Background Points [53, 58, 53, 56, 49, 47, 51, 52, 52, 51, 50, 51, 41, 48]
  Segment Points: [58, 54, 55, 149, 154, 155, 60, 53, 60, 60, 59, 60, 54, 52, 50]
  Otsu Point Segmentation
    Threshold: 148
    Segments: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match: L
    Match Error: 0
Reading char 4 @ (424, 47)
read_segments (424, 47)
  Background Points [48, 48, 42, 48, 46, 48, 48, 48, 45, 42, 46, 48, 36, 45]
  Segment Points: [150, 52, 53, 56, 57, 53, 50, 52, 161, 162, 48, 53, 49, 51, 51]
  Otsu Point Segmentation
    Threshold: 149
    Segments: [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    Best Match: T
    Match Error: 0
  Display: CPLT

  Reading frame 5...
Reading char 1 @ (35, 47)
read_segments (35, 47)
  Background Points [68, 64, 54, 57, 57, 54, 58, 61, 50, 59, 52, 61, 69, 65]
  Segment Points: [154, 66, 65, 148, 157, 159, 67, 65, 68, 64, 68, 66, 66, 63, 61]
  Otsu Point Segmentation
    Threshold: 147
    Segments: [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match: C
    Match Error: 0
Reading char 2 @ (165, 47)
read_segments (165, 47)
  Background Points [43, 57, 57, 61, 52, 65, 43, 50, 61, 43, 63, 51, 41, 50]
  Segment Points: [156, 151, 58, 59, 164, 161, 179, 167, 62, 62, 66, 65, 66, 64, 51]
  Otsu Point Segmentation
    Threshold: 150
    Segments: [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    Best Match: P
    Match Error: 0
Reading char 3 @ (295, 47)
read_segments (295, 47)
  Background Points [56, 51, 53, 53, 50, 60, 49, 58, 61, 50, 53, 53, 52, 45]
  Segment Points: [58, 56, 55, 151, 150, 156, 60, 54, 59, 60, 59, 60, 52, 53, 50]
  Otsu Point Segmentation
    Threshold: 149
    Segments: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match: L
    Match Error: 0
Reading char 4 @ (424, 47)
read_segments (424, 47)
  Background Points [47, 44, 44, 44, 51, 42, 51, 50, 48, 40, 47, 47, 43, 50]
  Segment Points: [150, 53, 54, 57, 56, 53, 50, 53, 160, 167, 50, 53, 52, 52, 49]
  Otsu Point Segmentation
    Threshold: 149
    Segments: [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    Best Match: T
    Match Error: 0
  Display: CPLT

  Reading frame 6...
Reading char 1 @ (35, 47)
read_segments (35, 47)
  Background Points [59, 60, 64, 62, 61, 58, 54, 56, 59, 59, 62, 58, 54, 55]
  Segment Points: [62, 66, 62, 63, 68, 65, 64, 65, 68, 65, 62, 65, 65, 61, 58]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match:
    Match Error: 0
Reading char 2 @ (165, 47)
read_segments (165, 47)
  Background Points [51, 62, 56, 57, 49, 56, 55, 54, 55, 52, 38, 57, 53, 57]
  Segment Points: [54, 57, 58, 61, 62, 64, 59, 58, 62, 60, 63, 62, 56, 64, 51]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match:
    Match Error: 0
Reading char 3 @ (295, 47)
read_segments (295, 47)
  Background Points [52, 55, 53, 56, 55, 58, 45, 54, 40, 48, 60, 48, 52, 46]
  Segment Points: [155, 152, 156, 55, 58, 59, 59, 57, 60, 62, 55, 57, 58, 56, 52]
  Otsu Point Segmentation
    Threshold: 151
    Segments: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match: 7
    Match Error: 0
Reading char 4 @ (424, 47)
read_segments (424, 47)
  Background Points [48, 49, 49, 39, 39, 49, 46, 36, 51, 37, 56, 50, 50, 51]
  Segment Points: [49, 152, 153, 57, 57, 54, 51, 52, 48, 56, 46, 55, 50, 55, 52]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match: 1
    Match Error: 0
  Display:   71

  Reading frame 7...
Reading char 1 @ (35, 47)
read_segments (35, 47)
  Background Points [48, 62, 58, 59, 51, 57, 60, 57, 57, 56, 53, 62, 54, 54]
  Segment Points: [64, 66, 65, 64, 67, 66, 66, 67, 68, 65, 63, 64, 65, 64, 61]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match:
    Match Error: 0
Reading char 2 @ (165, 47)
read_segments (165, 47)
  Background Points [50, 48, 59, 51, 40, 60, 45, 60, 45, 45, 54, 46, 40, 60]
  Segment Points: [57, 59, 59, 60, 63, 65, 60, 58, 58, 61, 63, 60, 56, 63, 50]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match:
    Match Error: 0
Reading char 3 @ (295, 47)
read_segments (295, 47)
  Background Points [53, 58, 56, 55, 57, 53, 39, 60, 48, 46, 56, 47, 49, 47]
  Segment Points: [156, 157, 156, 54, 59, 61, 58, 58, 61, 60, 57, 56, 61, 56, 51]
  Otsu Point Segmentation
    Threshold: 155
    Segments: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match: 7
    Match Error: 0
Reading char 4 @ (424, 47)
read_segments (424, 47)
  Background Points [42, 52, 27, 42, 44, 39, 39, 36, 36, 42, 49, 45, 50, 37]
  Segment Points: [49, 153, 151, 56, 58, 51, 52, 53, 48, 57, 49, 54, 51, 54, 51]
  Otsu Point Segmentation
    Threshold: 100
    Segments: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Best Match: 1
    Match Error: 0
  Display:   71

Frame Semantics:
  Frame 1:   71
    Unrecognized:   71
  Frame 2:  2.48
    Time: 02:48
  Frame 3:  2.48
    Time: 02:48
  Frame 4: CPLT
    State: Complete
  Frame 5: CPLT
    State: Complete
  Frame 6:   71
    Temperature: 71.0
  Frame 7:   71
    Unrecognized:   71
Published temperature: 71.0 °F to topic: home/kiln/temperature
Published time: 02:48 to topic: home/kiln/time
Published state: Complete to topic: home/kiln/state
state=Complete, temp=71.0, time=02:48
```


