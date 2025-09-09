from enum import Enum

class ComplianceState(Enum):
    """Enumeration for the possible PPE compliance states."""
    UNKNOWN = "UNKNOWN"
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"

class TrackedPerson:
    """
    Manages the compliance state for a single tracked person, ensuring that the
    status is "sticky" and persists across frames where the view is obscured.
    """
    def __init__(self, track_id: int, ppe_items: list[str]):
        """
        Initializes a TrackedPerson.

        Args:
            track_id (int): The unique ID assigned by the object tracker.
            ppe_items (list[str]): A list of PPE items to be monitored.
        """
        self.track_id = track_id
        # Initialize the status for each PPE item as UNKNOWN.
        self.ppe_status = {item: ComplianceState.UNKNOWN for item in ppe_items}

    def update_ppe_status(self, new_observations: dict[str, str]):
        """
        Updates the PPE status for each item based on new observations using the
        "sticky" state logic.

        Args:
            new_observations (dict[str, str]): A dictionary where keys are PPE
                items and values are the new observed states (e.g., 'COMPLIANT').
        """
        for item, observed_state_str in new_observations.items():
            if item not in self.ppe_status:
                continue  # Ignore observations for items not being tracked

            try:
                # Convert the string observation to a ComplianceState enum member.
                observed_state = ComplianceState(observed_state_str.upper())
            except ValueError:
                # If the verifier returns an invalid state, ignore it.
                continue

            current_state = self.ppe_status[item]

            # --- Sticky State Update Logic ---
            # 1. A known state (COMPLIANT or NON_COMPLIANT) is never overwritten
            #    by an UNKNOWN observation.
            if observed_state == ComplianceState.UNKNOWN:
                continue

            # 2. Any observation of COMPLIANT or NON_COMPLIANT should overwrite UNKNOWN.
            if current_state == ComplianceState.UNKNOWN:
                self.ppe_status[item] = observed_state
                continue

            # 3. A COMPLIANT state can only be overwritten by a NON_COMPLIANT observation.
            if current_state == ComplianceState.COMPLIANT:
                if observed_state == ComplianceState.NON_COMPLIANT:
                    self.ppe_status[item] = observed_state
                continue

            # 4. A NON_COMPLIANT state can be overwritten by a COMPLIANT observation
            #    (e.g., if the person puts on the required PPE).
            if current_state == ComplianceState.NON_COMPLIANT:
                if observed_state == ComplianceState.COMPLIANT:
                    self.ppe_status[item] = observed_state
                continue

    def get_status_dict(self) -> dict:
        """
        Returns the current compliance status as a JSON-serializable dictionary.
        """
        return {
            "track_id": self.track_id,
            "ppe_status": {item: state.value for item, state in self.ppe_status.items()}
        }

class StateManager:
    """
    Manages the lifecycle and state of all tracked persons in the video stream.
    """
    def __init__(self, ppe_items: list[str]):
        self.tracked_persons = {}  # A dictionary to hold TrackedPerson objects, keyed by track_id
        self.ppe_items = ppe_items

    def update_person_state(self, track_id: int, ppe_observations: dict[str, str]):
        """
        Updates the state of a specific person. If the person is new, they are
        added to the tracking system.
        """
        if track_id not in self.tracked_persons:
            # If this is a new person, create a new TrackedPerson instance.
            self.tracked_persons[track_id] = TrackedPerson(track_id, self.ppe_items)

        # Update the person's status with the latest observations.
        self.tracked_persons[track_id].update_ppe_status(ppe_observations)

    def get_all_statuses(self) -> list[dict]:
        """
        Returns a list of status dictionaries for all currently tracked persons.
        """
        return [person.get_status_dict() for person in self.tracked_persons.values()]

    def remove_stale_tracks(self, active_track_ids: set[int]):
        """
        Removes persons from the state manager that are no longer being tracked
        by the object tracker.
        """
        stale_ids = set(self.tracked_persons.keys()) - active_track_ids
        for track_id in stale_ids:
            del self.tracked_persons[track_id]
