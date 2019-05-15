from pedect.config.BasicConfig import BasicConfig
from pedect.tracker.FakeTracker import FakeTracker
from pedect.tracker.OpenCVTracker import OpenCVTracker

class TimesHolder:
    time0, time1, time2, time3, time4 = 0.0, 0.0, 0.0, 0.0, 0.0

    @staticmethod
    def getTimes():
        return TimesHolder.time0, TimesHolder.time1, TimesHolder.time2, TimesHolder.time3, TimesHolder.time4

def getTrackerFromConfig(config: BasicConfig):
    if config.trackerType == "re3":
        from pedect.tracker.Re3ObjectTracker import Re3ObjectTracker
        return Re3ObjectTracker()
    if config.trackerType == "fake":
        return FakeTracker()
    return OpenCVTracker(config.trackerType)