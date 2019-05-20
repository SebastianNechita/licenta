import time

from pedect.config.BasicConfig import BasicConfig
from pedect.predictor.Predictor import Predictor
from pedect.predictor.PredictedBox import PredictedBox
from pedect.predictor.VideoHolder import VideoHolder
from pedect.tracker.Tracker import Tracker
from pedect.utils.trackedObjectsOperations import *
from pedect.tracker.TimesHolder import TimesHolder as TH

class TrackerPredictor(Predictor):

    def __init__(self, predictor: Predictor, videoHolder: VideoHolder, tracker: Tracker, config: BasicConfig):
        self.predictor = predictor
        self.videoHolder = videoHolder
        self.tracker = tracker
        self.config = config
        self.activeObjects = {}

    def predictForFrame(self, frameNr: int):
        image = self.videoHolder.getFrame(frameNr)  # predictor.getFrame(frameNr)
        start = time.time()
        printd("Start A")
        imageHash = hash("%s-%s-%s-%s" % (self.videoHolder.chosenDataset, self.videoHolder.setName,
                                  self.videoHolder.videoNr, frameNr))
        self.activeObjects = refreshTrackedObjects(self.tracker, image, self.activeObjects, imageHash)
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
                                                            frameNr, probabilitiesDictionary, imageHash)
        # print(probabilitiesDictionary)
        printd("End D")

        TH.time3 += time.time() - start
        start = time.time()
        printd("Start E")

        self.activeObjects = removeOldObjects(self.activeObjects, frameNr, self.config.maxAge)
        printd("End E")

        TH.time4 += time.time() - start
        return [PredictedBox(int(v.getPos()[0] + 0.5), int(v.getPos()[1] + 0.5), int(v.getPos()[2] + 0.5),
                             int(v.getPos()[3] + 0.5), v.getLabel(), probabilitiesDictionary[k])
                for k, v in self.activeObjects.items()]

    # def finishPrediction(self):
    #     self.predictor.finishPrediction()
    #     self.tracker.clearTracker()
    #     self.activeObjects = {}


def printd(a):
    DEBUG = False
    if DEBUG:
        print(a)