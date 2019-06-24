from typing import Tuple

from tracker import re3_tracker

from pedect.tracker.Tracker import Tracker
from pedect.tracker.TimesHolder import TimesHolder


class Re3ObjectTracker(Tracker):

    # def parallelizable(self) -> bool:
    #     return False

    def __init__(self):
        Tracker.__init__(self)

    __tracker = re3_tracker.Re3Tracker()

    @staticmethod
    def getTracker():
        return Re3ObjectTracker.__tracker

    def track(self, uniqueId: str, image, bbox: Tuple[int, int, int, int] = None, imageHash: int = None) -> \
            Tuple[int, int, int, int]:
        TimesHolder.trackerAccessed += 1
        ans = tuple([int(round(x)) for x in self.__tracker.track(uniqueId, image, bbox)])
        return ans

