import copy
import random
from typing import Sequence

from tqdm import tqdm

from pedect.config.BasicConfig import BasicConfig
from pedect.evaluator.Evaluator import Evaluator
from pedect.predictor.GroundTruthPredictor import GroundTruthPredictor
from pedect.predictor.MinScoreWrapperPredictor import MinScoreWrapperPredictor
from pedect.predictor.TrackerPredictor import TrackerPredictor
from pedect.predictor.YoloPredictor import YoloPredictor
from pedect.tracker.Tracker import Tracker
from pedect.tracker.trackerHelper import getTrackerFromConfig
from pedect.utils.constants import MAX_VIDEO_LENGTH
import time
import pedect.utils.parallel
from pedect.utils.parallel import executor


class HyperParametersTuner:

    @staticmethod
    def __findAnswerForConfig(config, tracker, gtPredictors, maxFrames, withPartialOutput, threadNr) -> float:
        print("I was started! -----> I am thread " + str(threadNr))
        predictors = findTrackerPredictorsFromVideoList(tracker, config, gtPredictors)
        result = Evaluator(predictors, gtPredictors, maxFrames).evaluate(withPartialOutput)
        print(str(config.getTrackingHyperParameters()), " ---> ", result, threadNr)
        print("I am done! -------> I am thread " + str(threadNr))
        return result

    @staticmethod
    def __giveAns(config, tracker, gtPredictors, maxFrames, withPartialOutput, threadNr) -> float:
        print("I was started! -----> I am thread " + str(threadNr))
        predictors = findTrackerPredictorsFromVideoList(tracker, config, gtPredictors)
        result = Evaluator(predictors, gtPredictors, maxFrames).evaluate(withPartialOutput)
        result = 3
        print(str(config.getTrackingHyperParameters()), " ---> ", result)
        print("I am done! -------> I am thread " + str(threadNr))
        return result

    @staticmethod
    def tryToFindBestConfig(config, videosList, noIterations, ctRange, rtRange, stRange, smpRange, mspRange,
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
        executionList = []
        thrNo = 0
        for _ in range(noIterations):
            r = hpGenerator.getNextRange()
            newConfig = copy.deepcopy(config)
            newConfig.createThreshold, newConfig.removeThreshold, newConfig.surviveThreshold, \
                newConfig.surviveMovePercent, newConfig.minScorePrediction = r
            tracker = getTrackerFromConfig(newConfig)
            if not tracker.parallelizable():
                result = HyperParametersTuner.__findAnswerForConfig(newConfig, tracker, gtPredictors, maxFrames,
                                                                    withPartialOutput, thrNo)
                if result >= bestResult[1]:
                    bestResult = (newConfig, result)
            else:
                print("ans = ")
                # print(HyperParametersTuner.__findAnswerForConfig(newConfig, tracker, gtPredictors, maxFrames,
                #                                                     withPartialOutput, thrNo))
                # print(HyperParametersTuner.__giveAns())
                print(executor.submit(HyperParametersTuner.__giveAns, newConfig, tracker, gtPredictors, maxFrames,
                                      withPartialOutput, thrNo).result())
                # print(executor.submit(HyperParametersTuner.__findAnswerForConfig, newConfig,
                #                 tracker, gtPredictors, maxFrames, withPartialOutput, thrNo).result())
                # executionList.append((newConfig, executor.submit(HyperParametersTuner.__findAnswerForConfig, newConfig,
                #                                                  tracker, gtPredictors, maxFrames, withPartialOutput, thrNo)))
                print("Started a thread!")
            thrNo = thrNo + 1
        thrNo = 0
        for result in tqdm(executionList):
            print("Waiting for a thread! -> thread number " + str(thrNo))
            thrNo = thrNo + 1
            actualResult = result[1].result()
            print("Yaaaay we have a value!")
            if actualResult >= bestResult[1]:
                bestResult = (result[0], actualResult)

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
            result.append(random.randint(0, 100) / 100 * (cRange[1] - cRange[0]) + cRange[0])
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

