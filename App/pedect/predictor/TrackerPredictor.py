from pedect.config.BasicConfig import BasicConfig
from pedect.predictor.GroundTruthPredictor import GroundTruthPredictor
from pedect.predictor.Predictor import Predictor
from pedect.predictor.PredictedBox import PredictedBox
from pedect.tracker.Tracker import Tracker
from pedect.utils.trackedObjectsOperations import *


class TrackerPredictor(Predictor):
    def __init__(self, normalPredictor: Predictor, groundTruthPredictor: GroundTruthPredictor, tracker: Tracker,
                 config: BasicConfig):
        self.predictor = normalPredictor
        self.groundTruthPredictor = groundTruthPredictor
        self.tracker = tracker
        self.config = config
        self.activeObjects = {}


    def predictForFrame(self, frameNr: int):
        image = self.groundTruthPredictor.getFrame(frameNr)  # predictor.getFrame(frameNr)
        self.activeObjects = refreshTrackedObjects(self.tracker, image, self.activeObjects)
        predictedBBoxes = self.predictor.predictForFrame(frameNr)
        self.activeObjects, probabilitiesDictionary = moveOrDestroyTrackedObjects(self.activeObjects, predictedBBoxes,
                                                                                  self.config.surviveMovePercent,
                                                                                  self.config.surviveThreshold)
        self.activeObjects = createAndDestroyTrackedObjects(self.tracker, image, self.activeObjects, predictedBBoxes,
                                                            self.config.createThreshold, self.config.removeThreshold,
                                                            frameNr, probabilitiesDictionary)
        self.activeObjects = removeOldObjects(self.activeObjects, frameNr, self.config.maxAge)
        return [PredictedBox(int(v.getPos()[0] + 0.5), int(v.getPos()[1] + 0.5), int(v.getPos()[2] + 0.5),
                             int(v.getPos()[3] + 0.5), v.getLabel(), probabilitiesDictionary[k])
                for k, v in self.activeObjects.items()]




