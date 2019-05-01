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
        predictors = findTrackerPredictorsFromVideoList(tracker, config, gtPredictors)
        evaluator = Evaluator(predictors, gtPredictors, maxFrames)
        result = evaluator.evaluate(withPartialOutput)
        # print(str(config.getTrackingHyperParameters()), " ---> ", result, "; Thread number:", threadNr)
        return result

    # @staticmethod
    # def __giveAns(config, tracker, gtPredictors, maxFrames, withPartialOutput, threadNr) -> float:
    #     print("I was started! -----> I am thread " + str(threadNr))
    #     predictors = findTrackerPredictorsFromVideoList(tracker, config, gtPredictors)
    #     result = Evaluator(predictors, gtPredictors, maxFrames).evaluate(withPartialOutput)
    #     result = 3
    #     print(str(config.getTrackingHyperParameters()), " ---> ", result)
    #     print("I am done! -------> I am thread " + str(threadNr))
    #     return result

    @staticmethod
    def tryToFindBestConfig(config, videosList, noIterations, ctRange, rtRange, stRange, smpRange, mspRange,
                            maxFrames = MAX_VIDEO_LENGTH, withPartialOutput = False, rangeSize: int = None):
        gtPredictors = findGroundTruthFromVideoList(videosList)
        yoloPredictors = [YoloPredictor(gtPredictor, config) for gtPredictor in gtPredictors]
        result = Evaluator(yoloPredictors, gtPredictors, maxFrames).evaluate(withPartialOutput)
        print("YoloPredictor ", " ---> ", result)
        bestResult = (0, -1)
        if rangeSize is None:
            original = (config.createThreshold, config.removeThreshold, config.surviveThreshold, config.surviveMovePercent,
                        config.minScorePrediction)
            unaffected = (0.0, 0.0, 1.0, 1.0, 0.0)
            hpGenerator = HPGenerator([unaffected, original], [ctRange, rtRange, stRange, smpRange, mspRange])
        else:
            hpGenerator = GridHPGenerator([ctRange, rtRange, stRange, smpRange, mspRange], rangeSize)
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
                executionList.append((newConfig, executor.submit(HyperParametersTuner.__findAnswerForConfig, newConfig,
                                                                 tracker, gtPredictors, maxFrames, withPartialOutput, thrNo)))
                # print("Started a thread!")
            thrNo = thrNo + 1
        for i, result in tqdm(enumerate(executionList)):
            executionList[i] = result[0], result[1].result()
        executionList.sort(key=lambda x: (x[1]), reverse=True)
        executionList = executionList[:min(len(executionList), 100)]
        [print(str(config.getTrackingHyperParameters()), " ---> ", result) for config, result in executionList]
        return executionList[0]

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

class GridHPGenerator(HPGenerator):

    def __updateToRange(self, number, rng):
        return rng[0] + (rng[1] - rng[0]) * number

    def __init__(self, ranges, gridSize):
        assert gridSize > 1
        initialTries = []
        rn = [float(i / (gridSize - 1)) for i in range(gridSize)]
        emptyRange = [0.0]
        rns = []
        for i in range(5):
            if ranges[i][0] == ranges[i][1]:
                rns.append(emptyRange)
            else:
                rns.append(rn)
        for a in rns[0]:
            for b in rns[1]:
                for c in rns[2]:
                    for d in rns[3]:
                        for e in rns[4]:
                            a1 = self.__updateToRange(a, ranges[0])
                            b1 = self.__updateToRange(b, ranges[1])
                            c1 = self.__updateToRange(c, ranges[2])
                            d1 = self.__updateToRange(d, ranges[3])
                            e1 = self.__updateToRange(e, ranges[4])
                            configTry = (a1, b1, c1, d1, e1)
                            initialTries.append(configTry)
                            print(configTry)


        print(len(initialTries))
        super().__init__(initialTries, ranges)



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
    # print(predictors)
    # i = 0
    for gtPredictor in gtPredictors:
        # print("New iteration! -> ", gtPredictor)
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
    # print("Finished!")
    return predictors

