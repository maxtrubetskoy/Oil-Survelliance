import torch
import numpy as np
from ultralytics import YOLO

def load_person_detector(model_path: str):
    """
    Loads the YOLOv8 model from the specified file path and moves it to the
    appropriate device (GPU if available, otherwise CPU).

    Args:
        model_path (str): The path to the YOLOv8 model file (e.g., 'yolov8n.pt').

    Returns:
        YOLO: The loaded YOLO model object.
    """
    # Check for CUDA GPU availability and set the device accordingly.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading person detection model onto device: '{device}'")

    try:
        model = YOLO(model_path)
        model.to(device)
        print("Person detection model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real application, you might want to handle this more gracefully.
        raise

def detect_persons(model: YOLO, frame: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
    """
    Performs person detection on a single video frame using the loaded YOLO model.

    Args:
        model (YOLO): The loaded YOLO model object.
        frame (np.ndarray): The video frame (as a NumPy array) to process.
        confidence_threshold (float): The confidence score threshold for filtering
                                      detections.

    Returns:
        np.ndarray: A NumPy array containing the detections for persons. Each row
                    represents a detected person and is in the format:
                    [x1, y1, x2, y2, confidence]. Returns an empty array if no
                    persons are detected above the threshold.
    """
    # Run YOLO inference on the frame. `verbose=False` suppresses console output.
    results = model(frame, verbose=False)

    # The results object contains all detections. We need to filter these to get
    # only the persons (class ID 0 in the standard COCO dataset).
    person_detections = []

    # `results[0].boxes` contains the bounding box data for all detections.
    for box in results[0].boxes:
        # `box.cls` is the class ID of the detection.
        if int(box.cls) == 0:  # Class 0 corresponds to 'person'.
            # `box.conf` is the confidence score of the detection.
            confidence = float(box.conf)
            if confidence >= confidence_threshold:
                # `box.xyxy[0]` gives the bounding box coordinates as [x1, y1, x2, y2].
                # We convert it to a NumPy array for consistency.
                bbox = box.xyxy[0].cpu().numpy()
                person_detections.append(np.append(bbox, confidence))

    if not person_detections:
        # Return a correctly shaped empty array if no persons are found.
        return np.empty((0, 5))

    return np.array(person_detections)
