from typing import Tuple

import cv2

from pedect.tracker.Tracker import Tracker


class OpenCVTracker(Tracker):
    def parallelizable(self) -> bool:
        return True

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    def __init__(self, trackerType):
        self.trackerType = trackerType
        self.__trackers = {}

    def track(self, uniqueId: str, image, bbox: Tuple[int, int, int, int] = None) -> Tuple[int, int, int, int]:
        if bbox is not None:
            tr = self.OPENCV_OBJECT_TRACKERS[self.trackerType]()
            self.__trackers[uniqueId] = tr
            return tr.init(image, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))
        else:
            box = self.__trackers[uniqueId].update(image)[1]
            return box[0], box[1], box[2] + box[0], box[3] + box[1]

    def clearTracker(self):
        print("\nClearing " + str(len(self.__trackers)) + " trackers...")
        del self.__trackers
        self.__trackers = {}
