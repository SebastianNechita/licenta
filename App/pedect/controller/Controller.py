import os
from typing import Sequence, Tuple

from tqdm import tqdm

from pedect.config.BasicConfig import BasicConfig
from pedect.converter.ConverterToImages import ConverterToImagesYoloV3
from pedect.evaluator.Evaluator import Evaluator
from pedect.evaluator.HyperParametersTuner import HyperParametersTuner, findGroundTruthFromVideoList, \
    findTrackerPredictorsFromVideoList
from pedect.generator.NewDataGenerator import NewDataGenerator
from pedect.predictor.GroundTruthPredictor import GroundTruthPredictor
from pedect.predictor.TrackerPredictor import TrackerPredictor
from pedect.predictor.YoloPredictor import YoloPredictor
from pedect.tracker.Re3ObjectTracker import Re3ObjectTracker
from pedect.trainer.YoloTrainer import YoloTrainer
from pedect.utils.constants import MAX_VIDEO_LENGTH, ANNOTATIONS_FILE, DATA_DIR
from pedect.utils.demo import playVideo


class Controller:
    def __init__(self, config: BasicConfig) -> None:
        self.config = config
        self.imgSaveTextPattern = "%s-%s-%s-%s.jpg"
        self.tracker = Re3ObjectTracker.getTracker()
        self.trainer = YoloTrainer(config)
        self.converter = ConverterToImagesYoloV3(self.imgSaveTextPattern)

    def prepareTrainingSet(self, trainingList: Sequence[Tuple[str, str, str]] = None) -> None:
        if trainingList is None:
            trainingList = self.splitIntoBatches()[0]
        self.converter.clearDirectory()
        for video in trainingList:
            self.converter.saveImagesFromGroundTruth(video[0], video[1], video[2])
        self.converter.writeAnnotationsFile()

    def train(self) -> None:
        self.trainer.train()

    def evaluatePredictor(self, videosList: Sequence[Tuple[str, str, str]] = None, noFrames: int = MAX_VIDEO_LENGTH,
                          withPartialOutput: bool = False) -> float:
        if videosList is None:
            videosList = self.splitIntoBatches()[1]
        gtPredictors = findGroundTruthFromVideoList(videosList)
        yoloPredictors = [YoloPredictor(gtPredictor, self.config) for gtPredictor in gtPredictors]
        result = Evaluator(yoloPredictors, gtPredictors, noFrames).evaluate(withPartialOutput)
        return result

    def optimizeTrackerConfig(self, ctRange: Tuple[float, float], rtRange: Tuple[float, float],
                              stRange: Tuple[float, float], smpRange: Tuple[float, float], mspRange: Tuple[float, float],
                              videosList: Sequence[Tuple[str, str, str]] = None, noIterations: int = 100, noFrames: int=30,
                              withPartialOutput: bool = False) -> None:
        if videosList is None:
            videosList = self.splitIntoBatches()[2]
        bestConfig, result = HyperParametersTuner.tryToFindBestConfig(self.config, self.tracker, videosList,
                                                                      noIterations,
                                                                      ctRange, rtRange, stRange, smpRange, mspRange,
                                                                      noFrames, withPartialOutput)
        print("Found best config with MaP = %s:\n%s" % (result, str(bestConfig)))
        self.config = bestConfig

    def playVideo(self, video: Tuple[str, str, str], config: BasicConfig = None, nrFrames = MAX_VIDEO_LENGTH) -> None:
        if config is None:
            config = self.config
        gtPredictor = GroundTruthPredictor(video[0], video[1], video[2])
        yoloPredictor = YoloPredictor(gtPredictor, config)
        trackerPredictor = TrackerPredictor(yoloPredictor, gtPredictor, self.tracker, config)
        playVideo([(yoloPredictor, [0, 255, 0]), (gtPredictor, [255, 0, 0]), (trackerPredictor, [0, 0, 255])],
                  gtPredictor, nrFrames)

    def generateNewData(self, videoList: Sequence[Tuple[str, str, str]] = None, verbose: bool = False,
                        nrFrames: int = MAX_VIDEO_LENGTH) -> None:
        if videoList is None:
            videoList = self.splitIntoBatches()[3]

        NewDataGenerator.initializeDirectory(self.config.imageGenerationSavePath)
        gtPredictors = findGroundTruthFromVideoList(videoList)
        actualPredictors = findTrackerPredictorsFromVideoList(self.tracker, self.config, gtPredictors)
        toIterate = zip(gtPredictors, actualPredictors)
        if verbose:
            toIterate = tqdm(toIterate)
        for gtPredictor, actualPredictor in toIterate:
            generator = NewDataGenerator(actualPredictor, gtPredictor, self.imgSaveTextPattern)
            generator.generateNewData(self.config.imageGenerationSavePeriod, self.config.imageGenerationSavePath,
                                      self.config.imageGenerationSaveFileName, verbose, nrFrames)

    def retrain(self, trainId: str) -> None:
        self.config.trainId = trainId
        newAnnotationsFile = os.path.join(self.config.imageGenerationSavePath, self.config.imageGenerationSaveFileName)
        trainer = YoloTrainer(self.config, [ANNOTATIONS_FILE, newAnnotationsFile])
        trainer.train()

    @staticmethod
    def getAllVideoTuples(datasetName: str = None) -> Sequence[Tuple[str, str, str]]:
        if datasetName is None:
            datasetName = "caltech"
        datasetPath = os.path.join(DATA_DIR, datasetName)
        sets = os.listdir(datasetPath)
        sets = sorted([x for x in sets if "annotations" != x])
        result = []
        for videoSet in sets:
            if not os.path.isdir(os.path.join(datasetPath, videoSet)):
                continue
            videos = sorted(os.listdir(os.path.join(datasetPath, videoSet)))
            for videoName in videos:
                result.append((datasetName, str(videoSet), str(videoName.split(".seq")[0])))
        return result

    def splitIntoBatches(self) -> Tuple[Sequence[Tuple[str, str, str]], Sequence[Tuple[str, str, str]],
                                        Sequence[Tuple[str, str, str]], Sequence[Tuple[str, str, str]]]:
        aux = self.config.batchSplit
        splitPercent = [aux[0], aux[1], aux[2], aux[3]]
        sets = self.getAllVideoTuples()
        for i in range(len(splitPercent))[1:]:
            splitPercent[i] += splitPercent[i - 1]
        j = 0
        result = [[], [], [], []]
        for i in range(len(sets)):
            while i / len(sets) > splitPercent[j]:
                j = j + 1
            result[j].append(sets[i])
        return result[0], result[1], result[2], result[3]

    # a, b, c, d = controller.splitIntoBatches()

