from typing import Tuple

from pedect.config.BasicConfig import BasicConfig
from pedect.tracker.FakeTracker import FakeTracker
from pedect.tracker.OpenCVTracker import OpenCVTracker
from pedect.tracker.TimesHolder import TimesHolder
from pedect.tracker.Tracker import Tracker


def getNormalTrackerFromConfig(trackerType: str):
    if trackerType == "re3":
        from pedect.tracker.Re3ObjectTracker import Re3ObjectTracker
        return Re3ObjectTracker()
    if trackerType == "fake":
        return FakeTracker()
    return OpenCVTracker(trackerType)

def getTrackerFromTrackerType(trackerType):
    wordList = trackerType.split(' ')
    if len(wordList) == 1:
        return getNormalTrackerFromConfig(wordList[0])
    if wordList[0] != "cached":
        print("Error no such known tracker:", trackerType)
        exit(1)
    return CachingTracker(getNormalTrackerFromConfig(wordList[1]))

def getTrackerFromConfig(config: BasicConfig):
    return getTrackerFromTrackerType(config.trackerType)

class CachingTrackerManager:
    trackersConfigurations = {}

    @staticmethod
    def getConfigurationForTracker(trackerType: str):
        if trackerType not in CachingTrackerManager.trackersConfigurations:
            CachingTrackerManager.trackersConfigurations[trackerType] = ({}, {}, getTrackerFromTrackerType(trackerType))
        return CachingTrackerManager.trackersConfigurations[trackerType]

    @staticmethod
    def clearCacheForTrackerType(trackerType: str):
        CachingTrackerManager.trackersConfigurations.pop(trackerType)

class CachingTracker(Tracker):

    def __init__(self, tracker):
        Tracker.__init__(self)
        self.trackerType = "cached " + tracker.trackerType
        self.idsToHash, self.hashToAnswer, self.tracker = CachingTrackerManager.getConfigurationForTracker(tracker.trackerType)

    @staticmethod
    def __imageHash(image):
        return hash(str(image))

    def track(self, uniqueId: str, image, bbox: Tuple[int, int, int, int] = None, imageHash: int = None) -> \
            Tuple[int, int, int, int]:
        TimesHolder.cachedTrackerAccessed += 1
        if imageHash is None:
            print("Image hash is None!")
            imageHash = self.__imageHash(image)
        if bbox is not None:
            theHash = hash((imageHash, bbox))
            self.idsToHash[uniqueId] = theHash
            if theHash not in self.hashToAnswer:
                self.hashToAnswer[theHash] = {imageHash: self.tracker.track(str(theHash), image, bbox)}
            return self.hashToAnswer[theHash][imageHash]
        theHash = self.idsToHash[uniqueId]
        if imageHash not in self.hashToAnswer[theHash]:
            self.hashToAnswer[theHash][imageHash] = self.tracker.track(str(theHash), image)
        return self.hashToAnswer[theHash][imageHash]

    def parallelizable(self) -> bool:
        return False
