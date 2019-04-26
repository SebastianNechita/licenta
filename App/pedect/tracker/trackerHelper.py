from pedect.config.BasicConfig import BasicConfig
from pedect.tracker.OpenCVTracker import OpenCVTracker


def getTrackerFromConfig(config: BasicConfig):
    if config.trackerType == "re3":
        from pedect.tracker.Re3ObjectTracker import Re3ObjectTracker
        return Re3ObjectTracker()
    return OpenCVTracker(config.trackerType)