import time

from pedect.config.BasicConfig import BasicConfig
from pedect.predictor.Predictor import Predictor
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
        debug = False
        printd(frameNr, debug)
        image = self.videoHolder.getFrame(frameNr)  # predictor.getFrame(frameNr)
        start = time.time()
        printd("Start A", debug)
        imageHash = hash("%s-%s-%s-%s" % (self.videoHolder.chosenDataset, self.videoHolder.setName,
                                  self.videoHolder.videoNr, frameNr))
        self.activeObjects = refreshTrackedObjects(self.tracker, image, self.activeObjects, imageHash)
        printd(len(self.activeObjects), debug)
        printd("End A", debug)

        TH.time0 += time.time() - start
        start = time.time()
        printd("Start B", debug)
        predictedBBoxes = self.predictor.predictForFrame(frameNr)
        printd({k: str(v.getPos()) for k, v in self.activeObjects.items()}, debug)

        printd("End B", debug)

        TH.time1 += time.time() - start
        start = time.time()
        printd("Start C", debug)

        self.activeObjects, probabilitiesDictionary = moveOrDestroyTrackedObjects(self.activeObjects, predictedBBoxes,
                                                                                  self.config.surviveMovePercent,
                                                                                  self.config.surviveThreshold,
                                                                                  self.config.maxNrOfObjectsPerFrame)
        printd({k: str(v.getPos()) for k, v in self.activeObjects.items()}, debug)

        printd(len(self.activeObjects), debug)

        printd("End C", debug)

        TH.time2 += time.time() - start
        start = time.time()
        printd("Start D", debug)

        self.activeObjects = createAndDestroyTrackedObjects(self.tracker, image, self.activeObjects, predictedBBoxes,
                                                            self.config.createThreshold, self.config.removeThreshold,
                                                            frameNr, probabilitiesDictionary, imageHash)
        printd(len(self.activeObjects), debug)

        # print(probabilitiesDictionary)
        printd("End D", debug)

        TH.time3 += time.time() - start
        start = time.time()
        printd("Start E", debug)

        self.activeObjects = removeOldObjects(self.activeObjects, frameNr, self.config.maxAge)
        printd(len(self.activeObjects), debug)
        printd("MaxAge = %d" % self.config.maxAge, debug)
        printd("End E", debug)

        TH.time4 += time.time() - start
        return [PredictedBox(int(v.getPos()[0] + 0.5), int(v.getPos()[1] + 0.5), int(v.getPos()[2] + 0.5),
                             int(v.getPos()[3] + 0.5), v.getLabel(), probabilitiesDictionary[k])
                for k, v in self.activeObjects.items()]

    # def finishPrediction(self):
    #     self.predictor.finishPrediction()
    #     self.tracker.clearTracker()
    #     self.activeObjects = {}


def printd(a, debug):
    if debug:
        print(a)