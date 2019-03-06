from pedect.predictor.BasePredictor import BasePredictor
from pedect.predictor.PredictedBox import PredictedBox
from pedect.utils.trackedObjectsOperations import *


class TrackerPredictor(BasePredictor):
    def __init__(self, normalPredictor, groundTruthPredictor, tracker, config):
        self.predictor = normalPredictor
        self.groundTruthPredictor = groundTruthPredictor
        self.tracker = tracker
        self.config = config
        self.activeObjects = {}


    def predictForFrame(self, frameNr):
        image = self.groundTruthPredictor.getFrame(frameNr)  # predictor.getFrame(frameNr)
        self.activeObjects = refreshTrackedObjects(self.tracker, image, self.activeObjects)
        predictedBBoxes = self.predictor.predictForFrame(frameNr)
        self.activeObjects = moveOrDestroyTrackedObjects(self.activeObjects, predictedBBoxes,
                                                         self.config.surviveMovePercent, self.config.surviveThreshold)
        self.activeObjects = createAndDestroyTrackedObjects(self.tracker, image, self.activeObjects, predictedBBoxes,
                                                            self.config.createThreshold, self.config.removeThreshold,
                                                            frameNr)
        self.activeObjects = removeOldObjects(self.activeObjects, frameNr, self.config.maxAge)
        return [PredictedBox(int(v.getPos()[0] + 0.5), int(v.getPos()[1] + 0.5), int(v.getPos()[2] + 0.5),
                             int(v.getPos()[3] + 0.5), v.getLabel(), 1.0)
                for k, v in self.activeObjects.items()]




