from typing import Sequence

import cv2

from pedect.predictor.MinScoreWrapperPredictor import MinScoreWrapperPredictor
from pedect.predictor.PredictedBox import PredictedBox
from pedect.predictor.TrackerPredictor import TrackerPredictor
from pedect.utils.constants import MAX_VIDEO_LENGTH
from pedect.utils.trackedObjectsOperations import *

def addRectanglesToImage(image, predictedBBoxes: Sequence[PredictedBox], color: Sequence[int] = None, withSpace: bool = False):
    if color is None:
        color = [0, 255, 0]
    for predBox in predictedBBoxes:
        x1 = predBox.getX1()
        y1 = predBox.getY1()
        x2 = predBox.getX2() - x1
        y2 = predBox.getY2() - y1
        if withSpace:
            x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 + 2 , y2 + 2
        cv2.rectangle(image, (x1, y1, x2, y2), color)

def playVideo(predictors, videoHolder, noFrames = MAX_VIDEO_LENGTH):
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 640, 480)
    video = videoHolder.getVideo()
    for frameNr in range(min(videoHolder.getLength(), noFrames)):
        image = video[frameNr][:]  # predictor.getFrame(frameNr)
        for predictor, color in predictors:
            space = False
            if isinstance(predictor, MinScoreWrapperPredictor):
                space = True
            predictedBBoxes = predictor.predictForFrame(frameNr)
            addRectanglesToImage(image, predictedBBoxes, color, space)
        cv2.imshow('Video', image)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
