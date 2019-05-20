from typing import Tuple

from pedect.tracker.Tracker import Tracker
from pedect.tracker.TimesHolder import TimesHolder


class FakeTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.__trackers = {}
        self.trackerType = "fake"

    def track(self, uniqueId: str, image, bbox: Tuple[int, int, int, int] = None, imageHash: int = None) -> \
            Tuple[int, int, int, int]:
        TimesHolder.trackerAccessed += 1
        if bbox is not None:
            self.__trackers[uniqueId] = bbox
        else:
            return self.__trackers[uniqueId]

    def parallelizable(self) -> bool:
        return True