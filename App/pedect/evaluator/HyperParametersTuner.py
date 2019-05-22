import copy
import random
from typing import Sequence, Tuple

from tqdm import tqdm

from pedect.config.BasicConfig import BasicConfig
from pedect.evaluator.Evaluator import Evaluator, get_size, EvaluatorMemoryManager
from pedect.predictor.GroundTruthPredictor import GroundTruthPredictor
from pedect.predictor.MinScoreWrapperPredictor import MinScoreWrapperPredictor
from pedect.predictor.TrackerPredictor import TrackerPredictor
from pedect.predictor.YOLOPredictor import YOLOPredictor
from pedect.tracker.Tracker import Tracker
from pedect.tracker.trackerHelper import getTrackerFromConfig, CachingTrackerManager
from pedect.utils.constants import MAX_VIDEO_LENGTH
import time
from pedect.utils.parallel import executor
import gc


class HyperParametersTuner:

    @staticmethod
    def __findAnswerForConfig(config: BasicConfig, tracker: Tracker, gtPredictor: GroundTruthPredictor, evaluator: Evaluator, withPartialOutput: bool):
        predictor = YOLOPredictor(gtPredictor, config)
        predictor = TrackerPredictor(predictor, gtPredictor, tracker, config)
        predictor = MinScoreWrapperPredictor(predictor, config.minScorePrediction)
        evaluator.addEvaluation(predictor, gtPredictor, withPartialOutput)

    @staticmethod
    def tryToFindBestConfig(config: BasicConfig, videosList: Sequence[Tuple[str, str, str]], noIterations: int, trackerTypes: Sequence[str], ctRange: tuple, rtRange: tuple, stRange: tuple, smpRange: tuple, mspRange: tuple, maxFrames: int = MAX_VIDEO_LENGTH, withPartialOutput: bool = False, rangeSize: int = None):
        yoloEvaluator = Evaluator(maxFrames)
        if rangeSize is None:
            original = (config.trackerType, config.createThreshold, config.removeThreshold, config.surviveThreshold,
                        config.surviveMovePercent, config.minScorePrediction)
            unaffected = ("fake", 0.0, 0.0, 1.0, 1.0, 0.0)
            hpGenerator = HPGenerator([unaffected, original], [trackerTypes, ctRange, rtRange, stRange, smpRange, mspRange])
        else:
            hpGenerator = LinearHPGenerator([trackerTypes, ctRange, rtRange, stRange, smpRange, mspRange], rangeSize)
        executionList = []
        toIterate = range(noIterations)
        extraTracker = getTrackerFromConfig(config)
        evaluators = []
        configurations = []
        # trackers = []
        for _ in toIterate:
            r = hpGenerator.getNextRange()
            evaluators.append(Evaluator(maxFrames))
            newConfig = copy.deepcopy(config)
            newConfig.trackerType, newConfig.createThreshold, newConfig.removeThreshold, newConfig.surviveThreshold, \
            newConfig.surviveMovePercent, newConfig.minScorePrediction = r
            # tracker = getTrackerFromConfig(newConfig)
            configurations.append(newConfig)
            # trackers.append(tracker)
        # if not extraTracker.parallelizable():
        #     toIterate = tqdm(toIterate)
        for datasetName, setName, videoName in tqdm(videosList):
            gc.collect()
            gtPredictor = GroundTruthPredictor(datasetName, setName, videoName)
            yoloEvaluator.addEvaluation(YOLOPredictor(gtPredictor, config), gtPredictor, withPartialOutput)
            for i in tqdm(toIterate):
                newConfig = configurations[i]
                tracker = getTrackerFromConfig(newConfig)
                evaluator = evaluators[i]
                if not tracker.parallelizable():
                    if i > 0 and configurations[i - 1].trackerType != newConfig.trackerType:
                        CachingTrackerManager.clearCacheForTrackerType(configurations[i - 1].trackerType)
                    HyperParametersTuner.__findAnswerForConfig(newConfig, tracker, gtPredictor, evaluator,
                                                                        withPartialOutput)
                else:
                    executionList.append(executor.submit(HyperParametersTuner.__findAnswerForConfig, newConfig,
                                                                     tracker, gtPredictor, evaluator, withPartialOutput))
        result = yoloEvaluator.evaluate()
        print("YoloPredictor ", " ---> ", result)
        if extraTracker.parallelizable():
            for result in tqdm(executionList):
                result.result()
        executionList.clear()
        for i in tqdm(toIterate):
            executionList.append((configurations[i], evaluators[i].evaluate()))
        print("Memory in GB:", (get_size(EvaluatorMemoryManager.existentValuesReverse) + get_size(
            EvaluatorMemoryManager.existentValues) + get_size(evaluators)) / (1024 ** 3))
        executionList.sort(key=lambda x: (x[1]["mAP"]), reverse=True)
        # executionList = executionList[:min(len(executionList), 100)]
        # [print(str(config.getTrackingHyperParameters()), " ---> ", result) for config, result in executionList]
        return executionList

class HPGenerator:
    def __init__(self, initialTries: Sequence, ranges: Sequence[tuple]):
        self.initialTries = initialTries
        self.ranges = ranges

    def getNextRange(self):
        if len(self.initialTries) != 0:
            a = self.initialTries[0]
            self.initialTries = self.initialTries[1:]
            return a
        result = []
        for elem in self.ranges:
            if isinstance(elem, tuple):
                result.append(random.randint(0, 100) / 100 * (elem[1] - elem[0]) + elem[0])
            elif isinstance(elem, list):
                result.append(elem[random.randint(0, len(elem) - 1)])
            else:
                print("No known hyperParameter type to choose from!")
        return tuple(result)

class LinearHPGenerator(HPGenerator):

    def __updateToRange(self, number: float, rng: Tuple[float, float]):
        return int(100 * (rng[0] + (rng[1] - rng[0]) * number)) / 100.0

    def __init__(self, ranges: Sequence[tuple], gridSize: int):
        assert gridSize > 1
        initialTries = []
        rn = [float(i / (gridSize - 1)) for i in range(gridSize)]
        emptyRange = [0.0]
        rns = []
        for i in range(1, 6):
            if ranges[i][0] == ranges[i][1]:
                rns.append(emptyRange)
            else:
                rns.append(rn)
        for aa in ranges[0]:
            for a in rns[0]:
                for b in rns[1]:
                    for c in rns[2]:
                        for d in rns[3]:
                            for e in rns[4]:
                                a1 = self.__updateToRange(a, ranges[1])
                                b1 = self.__updateToRange(b, ranges[2])
                                c1 = self.__updateToRange(c, ranges[3])
                                d1 = self.__updateToRange(d, ranges[4])
                                e1 = self.__updateToRange(e, ranges[5])
                                configTry = (aa, a1, b1, c1, d1, e1)
                                initialTries.append(configTry)
                                # print(configTry)


        print(len(initialTries))
        super().__init__(initialTries, ranges)



def findGroundTruthFromVideoList(videosList: Sequence[Tuple[str, str, str]]) -> Sequence[GroundTruthPredictor]:
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
        yoloPredictor = YOLOPredictor(gtPredictor, config)
        # print("YoloPredictor ", time.time() - start)
        # start = time.time()
        trackerPredictor = TrackerPredictor(yoloPredictor, gtPredictor, tracker, config)
        # print("TrackerPredictor ", time.time() - start)
        # start = time.time()
        predictors.append(MinScoreWrapperPredictor(trackerPredictor, config.minScorePrediction))
        # print("MinScoreWrapperPredictor ", time.time() - start)
    # print("Finished!")
    return predictors



