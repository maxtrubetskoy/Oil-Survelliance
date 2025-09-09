# Real-time PPE Compliance Monitor

## 1. Objective

This project is a Python-based system designed to monitor Personal Protective Equipment (PPE) compliance in real-time. It processes video streams to detect individuals and verify if they are wearing the required PPE, such as helmets, protective glasses, and uniforms. The system is built to be efficient and intelligent, handling challenges like variable camera angles and temporary occlusions through a stateful, view-aware tracking pipeline.

## 2. Core Architecture

The system is built on a two-step, tracking-based pipeline to ensure both efficiency and accuracy.

1.  **Person Detection:** A high-performance YOLOv8 model identifies all persons in a given video frame. This is the most computationally intensive step and is run periodically (e.g., every N frames) to conserve resources.

2.  **Object Tracking:** Detected persons are passed to a SORT (Simple Online and Realtime Tracking) algorithm. The tracker is critical for:
    *   **Efficiency:** It uses a lightweight Kalman filter to track people across intermediate frames, avoiding the need to run the heavy person detector on every frame.
    *   **Identity Persistence:** Each person is assigned a persistent tracking ID, which is essential for monitoring their compliance status over time.

3.  **PPE Verification:** For each tracked person's bounding box, a specialized model (currently a placeholder) is run on the cropped image of that person to verify their PPE status.

### Stateful, View-Aware Compliance Logic

This is the core intelligence of the system. To avoid false alarms when PPE is temporarily obscured (e.g., a person turns their back to the camera), the system maintains a "sticky" state for each piece of PPE for every tracked person.

The status for any PPE item can be one of three states:

*   `COMPLIANT`: The PPE has been positively identified.
*   `NON_COMPLIANT`: The area where the PPE should be is clearly visible, and the item is confirmed to be missing.
*   `UNKNOWN`: The area is not visible or the view is otherwise obscured.

**State Update Rules:**
*   A person's status is initialized as `UNKNOWN`.
*   An observation of `COMPLIANT` or `NON_COMPLIANT` will always overwrite an `UNKNOWN` state.
*   An `UNKNOWN` observation will **not** overwrite a known `COMPLIANT` or `NON_COMPLIANT` state.
*   A `COMPLIANT` state persists until a clear `NON_COMPLIANT` observation is made.

## 3. Project Structure

```
/ppe-compliance-monitor
|-- main.py             # Main application entry point
|-- requirements.txt    # Project dependencies
|-- Dockerfile          # For containerized deployment
|-- README.md           # This file
|-- config/
|   |-- config.yaml     # Configuration for video streams, models, etc.
|-- src/
|   |-- person_detector.py # Loads and runs the YOLO person detector
|   |-- tracker.py        # Handles object tracking using SORT
|   |-- ppe_verifier.py   # Placeholder for the PPE verification model
|   |-- state_manager.py  # Manages the "sticky" state for each person
|-- models/
|   |-- yolov8n.pt      # Placeholder for the YOLOv8 model weights
|-- utils/
|   |-- video.py        # Utility functions for video processing
```

## 4. Setup and Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ppe-compliance-monitor
```

### Step 2: Install Dependencies

It is highly recommended to use a Python virtual environment to manage dependencies.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required Python packages from requirements.txt
pip install -r requirements.txt
```

### Step 3: Download the YOLOv8 Model

The person detection module requires the pre-trained weights for the YOLOv8 model. The current configuration is set to use `yolov8n.pt` (the nano version), which offers a good balance of speed and accuracy.

1.  Download the `yolov8n.pt` file from the official [YOLOv8 releases page](https://github.com/ultralytics/yolov8/releases).
2.  Place the downloaded `yolov8n.pt` file inside the `ppe-compliance-monitor/models/` directory, replacing the empty placeholder file.

## 5. Configuration

All runtime parameters are controlled by the `config/config.yaml` file.

*   **`video_stream`**: **You must change this path.** Set it to your video file (e.g., `"path/to/your/video.mp4"`) or the URL of a network video stream.
*   **`person_detector`**: Configure the model path (should be correct by default) and the confidence threshold for detections.
*   **`tracker`**: Adjust parameters for the SORT tracker, such as `max_age` (how long to track a lost object) and `min_hits` (how many detections to confirm a track).
*   **`ppe_items`**: A list of the PPE items you want the system to monitor.

## 6. How to Run

Once the setup is complete and the configuration is set, run the main application from the project's root directory:

```bash
python ppe-compliance-monitor/main.py
```

The application will start processing the video and will print structured JSON logs to the console for each frame.

## 7. Output Format

The application outputs a JSON object for each frame, detailing the compliance status of all currently tracked persons.

**Example Output:**
```json
{
    "frame_number": 150,
    "active_tracks": 2,
    "compliance_status": [
        {
            "track_id": 1,
            "ppe_status": {
                "helmet": "COMPLIANT",
                "glasses": "UNKNOWN",
                "breathing_device": "UNKNOWN",
                "uniform": "COMPLIANT"
            }
        },
        {
            "track_id": 2,
            "ppe_status": {
                "helmet": "NON_COMPLIANT",
                "glasses": "COMPLIANT",
                "breathing_device": "UNKNOWN",
                "uniform": "COMPLIANT"
            }
        }
    ]
}
```
