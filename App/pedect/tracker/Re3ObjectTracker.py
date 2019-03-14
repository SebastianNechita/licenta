from tracker import re3_tracker


class Re3ObjectTracker:
    def __init__(self):
        raise NotImplementedError

    __tracker = re3_tracker.Re3Tracker()

    @staticmethod
    def getTracker():
        return Re3ObjectTracker.__tracker

