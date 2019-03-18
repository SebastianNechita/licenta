from tracker import re3_tracker

from pedect.tracker.Tracker import Tracker


class Re3ObjectTracker(Tracker):
    def __init__(self):
        raise NotImplementedError

    __tracker = re3_tracker.Re3Tracker()

    @staticmethod
    def getTracker():
        return Re3ObjectTracker.__tracker

