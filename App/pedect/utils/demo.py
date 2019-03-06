import cv2
from pedect.utils.trackedObjectsOperations import *

def addRectanglesToImage(tracker, image, activeObjects, predictedBBoxes, createThreshold):
    for predBox in predictedBBoxes:
        if predBox.getProb() >= createThreshold:
            color = [0, 255, 0]
            box = predBox.getPos()
            cv2.rectangle(image, (box[0] - 1, box[1] - 1), (box[2] + 1, box[3] + 1), color, 2)

    for activeId, v in activeObjects.items():
        bbox = v.getPos()
        tracker.track(activeId, image, bbox)
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0, 0, 255], 2)


def playVideo(predictor, videoHolder, tracker, config):
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 640, 480)
    activeObjects = {}
    video = videoHolder.getVideo()
    for frameNr in range(videoHolder.getLength()):
        image = video[frameNr]  # predictor.getFrame(frameNr)
        activeObjects = refreshTrackedObjects(tracker, image, activeObjects)
        predictedBBoxes = predictor.predictForFrame(frameNr)
        activeObjects = moveOrDestroyTrackedObjects(activeObjects, predictedBBoxes, config.surviveMovePercent,
                                                    config.surviveThreshold)
        activeObjects = createAndDestroyTrackedObjects(tracker, image, activeObjects, predictedBBoxes,
                                                       config.createThreshold, config.removeThreshold, frameNr)
        addRectanglesToImage(tracker, image, activeObjects, predictedBBoxes, config.createThreshold)

        activeObjects = removeOldObjects(activeObjects, frameNr, config.maxAge)
        cv2.imshow('Video', image)
        cv2.waitKey(1)
    cv2.destroyAllWindows()