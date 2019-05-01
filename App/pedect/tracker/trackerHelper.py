from pedect.config.BasicConfig import BasicConfig
from pedect.tracker.FakeTracker import FakeTracker
from pedect.tracker.OpenCVTracker import OpenCVTracker


def getTrackerFromConfig(config: BasicConfig):
    if config.trackerType == "re3":
        from pedect.tracker.Re3ObjectTracker import Re3ObjectTracker
        return Re3ObjectTracker()
    if config.trackerType == "fake":
        return FakeTracker()
    return OpenCVTracker(config.trackerType)