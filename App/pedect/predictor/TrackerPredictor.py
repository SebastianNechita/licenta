import time

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
        self.times = (0, 0, 0, 0, 0)

    def predictForFrame(self, frameNr: int):
        t0, t1, t2, t3, t4 = self.times
        image = self.groundTruthPredictor.getFrame(frameNr)  # predictor.getFrame(frameNr)
        start = time.time()
        self.activeObjects = refreshTrackedObjects(self.tracker, image, self.activeObjects)
        t0 += time.time() - start
        start = time.time()
        predictedBBoxes = self.predictor.predictForFrame(frameNr)
        t1 += time.time() - start
        start = time.time()
        self.activeObjects, probabilitiesDictionary = moveOrDestroyTrackedObjects(self.activeObjects, predictedBBoxes,
                                                                                  self.config.surviveMovePercent,
                                                                                  self.config.surviveThreshold,
                                                                                  self.config.maxNrOfObjectsPerFrame)
        t2 += time.time() - start
        start = time.time()
        self.activeObjects = createAndDestroyTrackedObjects(self.tracker, image, self.activeObjects, predictedBBoxes,
                                                            self.config.createThreshold, self.config.removeThreshold,
                                                            frameNr, probabilitiesDictionary)
        t3 += time.time() - start
        start = time.time()
        self.activeObjects = removeOldObjects(self.activeObjects, frameNr, self.config.maxAge)
        t4 += time.time() - start
        # # # #
        # # # #
        # # # #
        # # # # sa nu uiti sa mai faci o chestie aici -> aia cu sa se propage schimbarile zise de reteaua neuronala
        # # # #
        # # # #
        # # # #
        self.times = (t0, t1, t2, t3, t4)
        return [PredictedBox(int(v.getPos()[0] + 0.5), int(v.getPos()[1] + 0.5), int(v.getPos()[2] + 0.5),
                             int(v.getPos()[3] + 0.5), v.getLabel(), probabilitiesDictionary[k])
                for k, v in self.activeObjects.items()]

    def finishPrediction(self):
        self.predictor.finishPrediction()
        self.tracker.clearTracker()
        self.groundTruthPredictor.finishPrediction()


