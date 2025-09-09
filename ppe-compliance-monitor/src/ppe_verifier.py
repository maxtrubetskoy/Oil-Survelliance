import random
import numpy as np

# The possible states that the PPE verifier can return for each item.
POSSIBLE_STATES = ["COMPLIANT", "NON_COMPLIANT", "UNKNOWN"]

def verify_ppe(cropped_person_image: np.ndarray, ppe_items: list[str]) -> dict[str, str]:
    """
    This is a placeholder function that simulates a real PPE verification model.

    In a production system, this function would run a trained deep learning
    model on the cropped image of a person to determine their PPE status. To
    allow for robust end-to-end testing of the pipeline, this function
    returns randomized observations.

    Args:
        cropped_person_image (np.ndarray): The cropped image of the person
            (this is not used in the placeholder but is part of the function
            signature for future integration).
        ppe_items (list[str]): The list of PPE items to be verified.

    Returns:
        dict[str, str]: A dictionary of observations, where keys are PPE items
                        and values are their observed states (e.g., 'COMPLIANT').
    """
    # This simulation is designed to be slightly more realistic than pure randomness.
    # It mimics a real-world scenario where some items are more likely to be
    # obscured (UNKNOWN) depending on the camera angle.
    observations = {}
    for item in ppe_items:
        if item in ['glasses', 'breathing_device']:
            # View-dependent items are more likely to be UNKNOWN.
            # Weights: [COMPLIANT, NON_COMPLIANT, UNKNOWN]
            state = random.choices(POSSIBLE_STATES, weights=[0.4, 0.1, 0.5], k=1)[0]
        else:
            # Items that are generally always visible, like helmets or uniforms.
            # Weights: [COMPLIANT, NON_COMPLIANT, UNKNOWN]
            state = random.choices(POSSIBLE_STATES, weights=[0.7, 0.2, 0.1], k=1)[0]
        observations[item] = state

    return observations
