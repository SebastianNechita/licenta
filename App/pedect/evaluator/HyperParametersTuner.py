import copy
import random
from typing import Sequence, Tuple

from numpy.ma import arange
from tqdm import tqdm

from pedect.config.BasicConfig import BasicConfig
from pedect.evaluator.Evaluator import Evaluator, get_size
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
    def tryToFindBestConfig(config: BasicConfig, videosList: Sequence[Tuple[str, str, str]], noIterations: int, trackerTypes: Sequence[str], ctRange: tuple, rtRange: tuple, stRange: tuple, smpRange: tuple, mspRange: tuple, maxFrames: int = MAX_VIDEO_LENGTH, withPartialOutput: bool = False, stepSize: float = None,
                            maxAgeRange: Sequence[int] = None):
        if maxAgeRange is None:
            maxAgeRange = [100]
        yoloEvaluator = Evaluator(maxFrames)
        if stepSize is None:
            original = (config.trackerType, config.createThreshold, config.removeThreshold, config.surviveThreshold,
                        config.surviveMovePercent, config.minScorePrediction)
            unaffected = ("fake", 0.0, 0.0, 1.0, 1.0, 0.0)
            hpGenerator = HPGenerator([unaffected, original], [trackerTypes, ctRange, rtRange, stRange, smpRange, mspRange, maxAgeRange], noIterations)
        else:
            hpGenerator = LinearHPGenerator([trackerTypes, ctRange, rtRange, stRange, smpRange, mspRange, maxAgeRange], stepSize, noIterations)
        executionList = []
        toIterate = range(hpGenerator.getNumberOfIterations())
        extraTracker = getTrackerFromConfig(config)
        evaluators = []
        configurations = []
        # trackers = []
        for _ in toIterate:
            r = hpGenerator.getNextRange()
            evaluators.append(Evaluator(maxFrames))
            newConfig = copy.deepcopy(config)
            newConfig.trackerType, newConfig.createThreshold, newConfig.removeThreshold, newConfig.surviveThreshold, \
            newConfig.surviveMovePercent, newConfig.minScorePrediction, newConfig.maxAge = r
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
            CachingTrackerManager.clearAllCache()
        result = yoloEvaluator.evaluate()
        print("YoloPredictor ", " ---> ", result)
        if extraTracker.parallelizable():
            for result in tqdm(executionList):
                result.result()
        executionList.clear()
        for i in tqdm(toIterate):
            executionList.append((configurations[i], evaluators[i].evaluate()))
        # print("Memory in GB:", (get_size(EvaluatorMemoryManager.existentValuesReverse) + get_size(
        #     EvaluatorMemoryManager.existentValues) + get_size(evaluators)) / (1024 ** 3))
        executionList.sort(key=lambda x: (x[1]["mAP"]), reverse=True)
        # executionList = executionList[:min(len(executionList), 100)]
        # [print(str(config.getTrackingHyperParameters()), " ---> ", result) for config, result in executionList]
        return executionList

class HPGenerator:
    def __init__(self, initialTries: Sequence, ranges: Sequence[tuple], noIterations: int):
        self.initialTries = initialTries
        self.ranges = ranges
        self.noIterations = noIterations if noIterations is not None else len(self.initialTries)

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

    def getNumberOfIterations(self):
        return self.noIterations

class LinearHPGenerator(HPGenerator):

    def __updateToRange(self, number: float, rng: Tuple[float, float]):
        return int(100 * (rng[0] + (rng[1] - rng[0]) * number)) / 100.0

    def __init__(self, ranges: Sequence[tuple], stepSize: float, noIterations: int):
        assert 0.0 < stepSize <= 1.0
        cnt = 20
        initialTries = []
        for aa in ranges[0]:
            for a in arange(ranges[1][0], ranges[1][1] + stepSize/2, stepSize):
                for b in arange(ranges[2][0], ranges[2][1] + stepSize/2, stepSize):
                    for c in arange(ranges[3][0], ranges[3][1] + stepSize/2, stepSize):
                        for d in arange(ranges[4][0], ranges[4][1] + stepSize/2, stepSize):
                            for e in arange(ranges[5][0], ranges[5][1] + stepSize/2, stepSize):
                                for f in ranges[6]:
                                    # a1 = self.__updateToRange(a, ranges[1])
                                    # b1 = self.__updateToRange(b, ranges[2])
                                    # c1 = self.__updateToRange(c, ranges[3])
                                    # d1 = self.__updateToRange(d, ranges[4])
                                    # e1 = self.__updateToRange(e, ranges[5])
                                    # configTry = (aa, a1, b1, c1, d1, e1)
                                    configTry = (aa, a, b, c, d, e, f)
                                    initialTries.append(configTry)
                                    cnt = cnt - 1
                                    if cnt >= 0:
                                        print(configTry)
                                    if cnt == 0:
                                        print("......")


        print(len(initialTries))
        super().__init__(initialTries, ranges, noIterations)



