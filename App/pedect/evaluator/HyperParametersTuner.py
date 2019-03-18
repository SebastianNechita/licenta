import copy
import random
from typing import Sequence

from tqdm import tqdm

from pedect.config.BasicConfig import BasicConfig
from pedect.evaluator.Evaluator import Evaluator
from pedect.predictor.GroundTruthPredictor import GroundTruthPredictor
from pedect.predictor.MinScoreWrapperPredictor import MinScoreWrapperPredictor
from pedect.predictor.Predictor import Predictor
from pedect.predictor.TrackerPredictor import TrackerPredictor
from pedect.predictor.YoloPredictor import YoloPredictor
from pedect.tracker.Tracker import Tracker
from pedect.utils.constants import MAX_VIDEO_LENGTH
import time


class HyperParametersTuner:

    @staticmethod
    def tryToFindBestConfig(config, tracker, videosList, noIterations, ctRange, rtRange, stRange, smpRange, mspRange,
                            maxFrames = MAX_VIDEO_LENGTH, withPartialOutput = False):
        gtPredictors = findGroundTruthFromVideoList(videosList)
        yoloPredictors = [YoloPredictor(gtPredictor, config) for gtPredictor in gtPredictors]
        result = Evaluator(yoloPredictors, gtPredictors, maxFrames).evaluate(withPartialOutput)
        print("YoloPredictor ", " ---> ", result)
        bestResult = (0, -1)
        original = (config.createThreshold, config.removeThreshold, config.surviveThreshold, config.surviveMovePercent,
                    config.minScorePrediction)
        unaffected = (0.0, 0.0, 1.0, 1.0, 0.0)
        hpGenerator = HPGenerator([unaffected, original], [ctRange, rtRange, stRange, smpRange, mspRange])
        for _ in tqdm(range(noIterations)):
            r = hpGenerator.getNextRange()
            newConfig = copy.deepcopy(config)
            newConfig.createThreshold, newConfig.removeThreshold, newConfig.surviveThreshold, \
                newConfig.surviveMovePercent, newConfig.minScorePrediction = r

            # print("Found new running configuration!")
            predictors = findTrackerPredictorsFromVideoList(tracker, newConfig, gtPredictors)
            # print("Found predictors!")

            result = Evaluator(predictors, gtPredictors, maxFrames).evaluate(withPartialOutput)
            print(str(newConfig.getTrackingHyperParameters()), " ---> ", result)
            if result >= bestResult[1]:
                bestResult = (newConfig, result)

        return bestResult

class HPGenerator:
    def __init__(self, initialTries, ranges):
        self.initialTries = initialTries
        self.ranges = ranges

    def getNextRange(self):
        if len(self.initialTries) != 0:
            a = self.initialTries[0]
            self.initialTries = self.initialTries[1:]
            return a
        result = []
        for cRange in self.ranges:
            result.append(random.random() * (cRange[1] - cRange[0]) + cRange[0])
        return tuple(result)



def findGroundTruthFromVideoList(videosList: Sequence[tuple]) -> Sequence[GroundTruthPredictor]:
    gtPredictors = []
    # i = 0
    for datasetName, setName, videoName in videosList:
        # print(i)
        # i = i + 1
        start = time.time()
        # print("Start ")
        gtPredictor = GroundTruthPredictor(datasetName, setName, videoName)
        # print("GTPredictor ", time.time() - start)
        gtPredictors.append(gtPredictor)
    return gtPredictors

def findTrackerPredictorsFromVideoList(tracker: Tracker, config: BasicConfig,
                                       gtPredictors: Sequence[GroundTruthPredictor]) -> Sequence[MinScoreWrapperPredictor]:
    predictors = []
    # i = 0
    for gtPredictor in gtPredictors:
        # print(i)
        # i = i + 1
        # print("Start ")
        # start = time.time()
        yoloPredictor = YoloPredictor(gtPredictor, config)
        # print("YoloPredictor ", time.time() - start)
        # start = time.time()
        trackerPredictor = TrackerPredictor(yoloPredictor, gtPredictor, tracker, config)
        # print("TrackerPredictor ", time.time() - start)
        # start = time.time()
        predictors.append(MinScoreWrapperPredictor(trackerPredictor, config.minScorePrediction))
        # print("MinScoreWrapperPredictor ", time.time() - start)

    return predictors

