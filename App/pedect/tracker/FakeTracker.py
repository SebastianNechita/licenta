from typing import Tuple

from pedect.tracker.Tracker import Tracker


class FakeTracker(Tracker):
    def __init__(self):
        self.__trackers = {}

    def track(self, uniqueId: str, image, bbox: Tuple[int, int, int, int] = None) -> Tuple[int, int, int, int]:
        if bbox is not None:
            self.__trackers[uniqueId] = bbox
        else:
            return self.__trackers[uniqueId]

    def parallelizable(self) -> bool:
        return True