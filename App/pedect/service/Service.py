from typing import *

from pedect.converter.ConverterToImages import ConverterToImagesYOLOv3
from pedect.evaluator.HyperParametersTuner import *
from pedect.generator.NewDataGenerator import NewDataGenerator
from pedect.predictor.GroundTruthPredictor import GroundTruthPredictor
from pedect.predictor.TrackerPredictor import TrackerPredictor
from pedect.predictor.YOLOPredictor import YOLOPredictor
from pedect.tracker.trackerHelper import getTrackerFromConfig, CachingTrackerManager
from pedect.trainer.YOLOTrainer import YOLOTrainer
from pedect.utils.constants import *
from pedect.utils.demo import playVideo
from pedect.tracker.TimesHolder import TimesHolder as TH
from pedect.utils.osUtils import emptyDirectory


class Service:
    def __init__(self, config: BasicConfig) -> None:
        self.config = config
        self.imgSaveTextPattern = "%s-%s-%s-%s.jpg"
        self.tracker = getTrackerFromConfig(config)
        self.trainer = YOLOTrainer(config)
        self.converter = ConverterToImagesYOLOv3(self.imgSaveTextPattern)

    def prepareTrainingSet(self, trainingList: Sequence[Tuple[str, str, str]] = None) -> None:
        if trainingList is None:
            trainingList = self.splitIntoBatches()[0]
        print("Training list is ", trainingList)
        self.converter.clearDirectory()
        for video in trainingList:
            self.converter.saveImagesFromGroundTruth(video[0], video[1], video[2])
        self.converter.writeAnnotationsFile()

    def train(self) -> None:
        self.trainer.train()

    def evaluatePredictor(self, videosList: Sequence[Tuple[str, str, str]] = None, noFrames: int = MAX_VIDEO_LENGTH, withPartialOutput: bool = False) -> float:
        if videosList is None:
            videosList = self.splitIntoBatches()[1]
        gtPredictors = findGroundTruthFromVideoList(videosList)
        yoloPredictors = [YOLOPredictor(gtPredictor, self.config) for gtPredictor in gtPredictors]
        result = Evaluator(yoloPredictors, gtPredictors, noFrames).evaluate(withPartialOutput)
        return result

    def optimizeTrackerConfig(self, ctRange: Tuple[float, float], rtRange: Tuple[float, float], stRange: Tuple[float, float], smpRange: Tuple[float, float], mspRange: Tuple[float, float], videosList: Sequence[Tuple[str, str, str]] = None, noIterations: int = 100, noFrames: int=30, withPartialOutput: bool = False, rangeSize: int = None) -> None:
        if videosList is None:
            videosList = self.splitIntoBatches()[2]
        print("Working on ", videosList)
        trackers = ["cached medianflow", "cached mosse", "cached kcf"]
        baseDir = 'results'
        keys = ["createThreshold", "removeThreshold", "surviveThreshold", "surviveMovePercent", "minScorePrediction"]
        initialTrackerType = self.config.trackerType
        emptyDirectory(baseDir)
        for trackerType in trackers:
            print("Tracker type: ", trackerType)
            self.config.trackerType = trackerType
            results = HyperParametersTuner.tryToFindBestConfig(self.config, videosList,
                                                               noIterations,
                                                               ctRange, rtRange, stRange, smpRange, mspRange,
                                                               noFrames, withPartialOutput, rangeSize)
            words = trackerType.split(" ")
            if len(words) > 1:
                CachingTrackerManager.clearCacheForTrackerType(words[1])
            f = open(os.path.join(baseDir, trackerType.replace(" ", "_") + ".txt"), "w")
            f.write('{:18s} | {:18s} | {:18s} | {:18s} | {:18s} | Result \n'.format(*keys))
            for result in results:
                params = ([result[0].getTrackingHyperParameters()[x] for x in keys] + [result[1]])
                # print(params)
                f.write('{:18f} | {:18f} | {:18f} | {:18f} | {:18f} | {:18f}\n'.format(*params))
            f.close()
        self.config.trackerType = initialTrackerType
        # print("Found best config with MaP = %s:\n%s" % (result, str(bestConfig)))
        # self.config = bestConfig

    def playVideo(self, video: Tuple[str, str, str], config: BasicConfig = None, nrFrames = MAX_VIDEO_LENGTH) -> None:
        if config is None:
            config = self.config
        gtPredictor = GroundTruthPredictor(video[0], video[1], video[2])
        yoloPredictor = YOLOPredictor(gtPredictor, config)
        trackerPredictor = TrackerPredictor(yoloPredictor, gtPredictor, self.tracker, config)
        playVideo([(yoloPredictor, [0, 255, 0]), (gtPredictor, [255, 0, 0]), (trackerPredictor, [0, 0, 255])],
                  gtPredictor, nrFrames)

    def generateNewData(self, videoList: Sequence[Tuple[str, str, str]] = None, verbose: bool = False, nrFrames: int = MAX_VIDEO_LENGTH) -> None:
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
        trainer = YOLOTrainer(self.config, [ANNOTATIONS_FILE, newAnnotationsFile])
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
        return sorted(list(set(result)))

    def splitIntoBatches(self) -> Tuple[Sequence[Tuple[str, str, str]], Sequence[Tuple[str, str, str]], Sequence[Tuple[str, str, str]], Sequence[Tuple[str, str, str]]]:
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

    @staticmethod
    def getRunningTimesPercentForTracker():
        times = list(TH.getTimes())
        stimes = sum(times)
        return [int(100 * x / stimes) / 100 for x in times]


