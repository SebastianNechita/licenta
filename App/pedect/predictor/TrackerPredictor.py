import time

from constants import LOG_DIR

from pedect.config.BasicConfig import BasicConfig
from pedect.predictor.GroundTruthPredictor import GroundTruthPredictor
from pedect.predictor.Predictor import Predictor
from pedect.predictor.PredictedBox import PredictedBox
from pedect.tracker.Tracker import Tracker
from pedect.utils.trackedObjectsOperations import *
from pedect.tracker.trackerHelper import TimesHolder as TH

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
        start = time.time()
        printd("Start A")
        self.activeObjects = refreshTrackedObjects(self.tracker, image, self.activeObjects)
        printd("End A")

        TH.time0 += time.time() - start
        start = time.time()
        printd("Start B")
        predictedBBoxes = self.predictor.predictForFrame(frameNr)
        printd("End B")

        TH.time1 += time.time() - start
        start = time.time()
        printd("Start C")

        self.activeObjects, probabilitiesDictionary = moveOrDestroyTrackedObjects(self.activeObjects, predictedBBoxes,
                                                                                  self.config.surviveMovePercent,
                                                                                  self.config.surviveThreshold,
                                                                                  self.config.maxNrOfObjectsPerFrame)
        printd("End C")

        TH.time2 += time.time() - start
        start = time.time()
        printd("Start D")

        self.activeObjects = createAndDestroyTrackedObjects(self.tracker, image, self.activeObjects, predictedBBoxes,
                                                            self.config.createThreshold, self.config.removeThreshold,
                                                            frameNr, probabilitiesDictionary)
        printd("End D")

        TH.time3 += time.time() - start
        start = time.time()
        printd("Start E")

        self.activeObjects = removeOldObjects(self.activeObjects, frameNr, self.config.maxAge)
        printd("End E")

        TH.time4 += time.time() - start
        # # # #
        # # # #
        # # # #
        # # # # TODO: sa nu uiti sa mai faci o chestie aici -> aia cu sa se propage schimbarile zise de reteaua neuronala
        # # # #
        # # # #
        # # # #
        return [PredictedBox(int(v.getPos()[0] + 0.5), int(v.getPos()[1] + 0.5), int(v.getPos()[2] + 0.5),
                             int(v.getPos()[3] + 0.5), v.getLabel(), probabilitiesDictionary[k])
                for k, v in self.activeObjects.items()]

    def finishPrediction(self):
        self.predictor.finishPrediction()
        self.tracker.clearTracker()
        self.groundTruthPredictor.finishPrediction()


def printd(a):
    DEBUG = False
    if DEBUG:
        print(a)