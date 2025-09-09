import yaml
import json
import numpy as np
import sys
from pathlib import Path

# Add the project root to the Python path to allow for absolute imports
# This makes the script runnable from anywhere
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.person_detector import load_person_detector, detect_persons
from src.tracker import Sort
from src.state_manager import StateManager
from src.ppe_verifier import verify_ppe
from utils.video import read_video_frames, crop_bbox_from_frame

def main():
    """
    Main function to run the entire PPE compliance monitoring pipeline.
    It loads configuration, initializes all modules, and processes a video
    stream frame by frame.
    """
    # 1. Load Configuration from YAML file
    print("Loading configuration...")
    config_path = project_root / "config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    video_path = config['video_stream']
    person_detector_config = config['person_detector']
    tracker_config = config['tracker']
    run_detection_every_n_frames = config['run_detection_every_n_frames']
    ppe_items = config['ppe_items']

    # Resolve the model path relative to the project root
    person_detector_config['model_path'] = str(project_root / person_detector_config['model_path'])

    # 2. Initialize Core Components
    print("Initializing core components...")
    person_model = load_person_detector(person_detector_config['model_path'])
    tracker = Sort(
        max_age=tracker_config['max_age'],
        min_hits=tracker_config['min_hits'],
        iou_threshold=tracker_config['iou_threshold']
    )
    state_manager = StateManager(ppe_items=ppe_items)

    # 3. Start the Main Video Processing Loop
    print(f"Starting video processing for: {video_path}")
    # Note: The user needs to provide a real video file path in config.yaml
    frame_generator = read_video_frames(video_path)

    for frame_num, frame in enumerate(frame_generator):

        # To improve performance, run the heavy person detector periodically.
        if frame_num % run_detection_every_n_frames == 0:
            detections = detect_persons(
                person_model,
                frame,
                confidence_threshold=person_detector_config['confidence_threshold']
            )
        else:
            # In between detection frames, provide an empty array to the tracker,
            # which will rely on its Kalman filter to predict movement.
            detections = np.empty((0, 5))

        # Update the tracker with the latest detections to get tracked objects.
        # The output is an array of [x1, y1, x2, y2, track_id].
        tracked_objects = tracker.update(detections)

        active_track_ids = set()

        # Process each currently tracked object
        if tracked_objects.size > 0:
            for track in tracked_objects:
                bbox = track[:4]
                track_id = int(track[4])
                active_track_ids.add(track_id)

                # Crop the person's bounding box from the frame.
                cropped_person = crop_bbox_from_frame(frame, bbox)

                # Skip verification if the cropped image is invalid (e.g., zero size).
                if cropped_person.size == 0:
                    continue

                # Get PPE observations from the (placeholder) verifier.
                ppe_observations = verify_ppe(cropped_person, ppe_items)

                # Update the person's compliance state using the sticky logic.
                state_manager.update_person_state(track_id, ppe_observations)

        # Clean up the state manager by removing tracks that are no longer active.
        state_manager.remove_stale_tracks(active_track_ids)

        # Get the latest status of all tracked persons.
        all_statuses = state_manager.get_all_statuses()

        # 4. Print Structured JSON Log Output for the current frame
        log_output = {
            "frame_number": frame_num,
            "active_tracks": len(all_statuses),
            "compliance_status": all_statuses
        }
        print(json.dumps(log_output, indent=2))

    print("\nVideo processing finished.")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[ERROR] A file was not found. Please check your paths in config/config.yaml.")
        print(f"Details: {e}")
    except Exception as e:
        # A broad exception handler to catch other potential errors during setup or processing.
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        print("Please ensure all dependencies are installed correctly and model files are in place.")
