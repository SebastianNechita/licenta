import os
from typing import Sequence

from tqdm import tqdm

from pedect.config.BasicConfig import BasicConfig
from pedect.converter.ConverterToImages import ConverterToImagesYoloV3
from pedect.evaluator.HyperParametersTuner import HyperParametersTuner, findGroundTruthFromVideoList, \
    findTrackerPredictorsFromVideoList
from pedect.generator.NewDataGenerator import NewDataGenerator
from pedect.predictor.GroundTruthPredictor import GroundTruthPredictor
from pedect.predictor.TrackerPredictor import TrackerPredictor
from pedect.predictor.YoloPredictor import YoloPredictor
from pedect.tracker.Re3ObjectTracker import Re3ObjectTracker
from pedect.trainer.YoloTrainer import YoloTrainer
from pedect.utils.constants import MAX_VIDEO_LENGTH, ANNOTATIONS_FILE
from pedect.utils.demo import playVideo


class Controller:
    def __init__(self, config: BasicConfig) -> None:
        self.config = config
        self.imgSaveTextPattern = "%s-%s-%s-%s.jpg"
        self.tracker = Re3ObjectTracker.getTracker()
        self.trainer = YoloTrainer(config)
        self.converter = ConverterToImagesYoloV3(self.imgSaveTextPattern)

    def prepareTrainingSet(self, trainingList: Sequence[tuple]) -> None:
        self.converter.clearDirectory()
        for video in trainingList:
            self.converter.saveImagesFromGroundTruth(video[0], video[1], video[2])
        self.converter.writeAnnotationsFile()

    def train(self) -> None:
        self.trainer.train()

    def optimizeTrackerConfig(self, ctRange: tuple, rtRange: tuple, stRange: tuple, smpRange: tuple, mspRange: tuple,
                              videosList: list, noIterations: int = 100, noFrames: int=30,
                              withPartialOutput: bool=False) -> None:
        bestConfig, result = HyperParametersTuner.tryToFindBestConfig(self.config, self.tracker, videosList,
                                                                      noIterations,
                                                                      ctRange, rtRange, stRange, smpRange, mspRange,
                                                                      noFrames, withPartialOutput)
        print("Found best config with MaP = %s:\n%s" % (result, str(bestConfig)))
        self.config = bestConfig

    def playVideo(self, video: tuple, config: BasicConfig = None, nrFrames = MAX_VIDEO_LENGTH) -> None:
        if config is None:
            config = self.config
        gtPredictor = GroundTruthPredictor(video[0], video[1], video[2])
        yoloPredictor = YoloPredictor(gtPredictor, config)
        trackerPredictor = TrackerPredictor(yoloPredictor, gtPredictor, self.tracker, config)
        playVideo([(yoloPredictor, [0, 255, 0]), (gtPredictor, [255, 0, 0]), (trackerPredictor, [0, 0, 255])],
                  gtPredictor, nrFrames)

    def generateNewData(self, videoList: Sequence[tuple], verbose: bool = False, nrFrames: int = MAX_VIDEO_LENGTH) \
            -> None:
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
