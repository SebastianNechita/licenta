from pedect.config.BasicConfig import BasicConfig
from pedect.converter.ConverterToImages import ConverterToImagesYoloV3
from pedect.evaluator.HyperParametersTuner import HyperParametersTuner
from pedect.predictor.GroundTruthPredictor import GroundTruthPredictor
from pedect.predictor.TrackerPredictor import TrackerPredictor
from pedect.predictor.YoloPredictor import YoloPredictor
from pedect.tracker.Re3ObjectTracker import Re3ObjectTracker
from pedect.trainer.YoloTrainer import YoloTrainer
from pedect.utils.demo import playVideo


class Controller:
    def __init__(self, config):
        self.config = config
        self.imgSaveTextPattern = "%s-%s-%s-%s.jpg"
        self.tracker = Re3ObjectTracker.getTracker()
        self.trainer = YoloTrainer(config)
        self.converter = ConverterToImagesYoloV3(self.imgSaveTextPattern)

    def prepareTrainingSet(self, trainingList: list):
        self.converter.clearDirectory()
        for video in trainingList:
            self.converter.saveImagesFromGroundTruth(video[0], video[1], video[2])
        self.converter.writeAnnotationsFile()

    def train(self):
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

    def playVideo(self, video: tuple, config: BasicConfig = -1):
        if config == -1:
            config = self.config
        gtPredictor = GroundTruthPredictor(video[0], video[1], video[2])
        yoloPredictor = YoloPredictor(gtPredictor, config)
        # trackerPredictor = TrackerPredictor(yoloPredictor, gtPredictor, self.tracker, config)
        playVideo(yoloPredictor, gtPredictor, self.tracker, config)
