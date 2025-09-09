import cv2
import numpy as np

def read_video_frames(video_path: str):
    """
    Creates a generator that reads and yields frames from a video file one by one.

    This approach is memory-efficient as it doesn't load the entire video into
    memory. It also ensures that the video capture object is released properly.

    Args:
        video_path (str): The path to the video file.

    Yields:
        np.ndarray: The next frame from the video as a NumPy array (in BGR format,
                    as read by OpenCV).

    Raises:
        FileNotFoundError: If the video file cannot be opened.
    """
    # Create a VideoCapture object.
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully.
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Could not open video file at '{video_path}'")

    try:
        while True:
            # Read one frame from the video.
            ret, frame = cap.read()

            # If `ret` is False, it means we have reached the end of the video.
            if not ret:
                break

            yield frame
    finally:
        # Ensure the video capture is released to free up resources.
        cap.release()

def crop_bbox_from_frame(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Safely crops a region defined by a bounding box from a video frame.

    This function handles cases where the bounding box coordinates might be
    outside the frame's dimensions by clipping them to the valid range.

    Args:
        frame (np.ndarray): The video frame from which to crop.
        bbox (np.ndarray): A NumPy array representing the bounding box in the
                           format [x1, y1, x2, y2].

    Returns:
        np.ndarray: The cropped region of the frame as a new NumPy array.
    """
    # Convert bounding box coordinates to integers.
    x1, y1, x2, y2 = bbox.astype(int)

    # Get the dimensions of the frame.
    frame_h, frame_w, _ = frame.shape

    # Clip the coordinates to ensure they are within the frame's boundaries.
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    # Return the cropped region.
    return frame[y1:y2, x1:x2]
