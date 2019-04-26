from typing import Tuple

from tracker import re3_tracker

from pedect.tracker.Tracker import Tracker


class Re3ObjectTracker(Tracker):
    def parallelizable(self) -> bool:
        return False

    def __init__(self):
        raise NotImplementedError

    __tracker = re3_tracker.Re3Tracker()

    @staticmethod
    def getTracker():
        return Re3ObjectTracker.__tracker

    def track(self, uniqueId: str, image, bbox: Tuple[int, int, int, int] = None) -> Tuple[int, int, int, int]:
        return self.__tracker.track(uniqueId, image, bbox)

