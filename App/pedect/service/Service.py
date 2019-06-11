from pedect.config.BasicConfig import saveConfiguration
from pedect.converter.ConverterToImages import ConverterToImagesYOLOv3
from pedect.evaluator.HyperParametersTuner import *
from pedect.generator.NewDataGenerator import NewDataGenerator
from pedect.predictor.GroundTruthPredictor import GroundTruthPredictor
from pedect.predictor.TrackerPredictor import TrackerPredictor
from pedect.predictor.YOLOPredictor import YOLOPredictor
from pedect.tracker.trackerHelper import getTrackerFromConfig
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
        self.converter = ConverterToImagesYOLOv3(self.imgSaveTextPattern)

    def getTrainingVideoList(self):
        return self.splitIntoBatches()[0]

    def getEvaluationVideoList(self):
        return self.splitIntoBatches()[1]

    def getTuningVideoList(self):
        return self.splitIntoBatches()[2]

    def getGenerationVideoList(self):
        return self.splitIntoBatches()[3]

    def prepareTrainingSet(self, trainingList: Sequence[Tuple[str, str, str]] = None) -> None:
        if trainingList is None:
            trainingList = self.getTrainingVideoList()
        print("Training list is ", trainingList)
        self.converter.clearDirectory()
        for video in trainingList:
            self.converter.saveImagesFromGroundTruth(video[0], video[1], video[2])
        self.converter.writeAnnotationsFile()

    def train(self, config) -> None:
        trainer = YOLOTrainer(config)
        trainer.train()

    def evaluatePredictor(self, config, videosList: Sequence[Tuple[str, str, str]] = None, noFrames: int = MAX_VIDEO_LENGTH, withPartialOutput: bool = False) -> dict:
        if videosList is None:
            videosList = self.getEvaluationVideoList()
        evaluator = Evaluator(noFrames)
        for datasetName, setName, videoName in videosList:
            gtPredictor = GroundTruthPredictor(datasetName, setName, videoName)
            yoloPredictor = YOLOPredictor(gtPredictor, config)
            evaluator.addEvaluation(yoloPredictor, gtPredictor, withPartialOutput)
        result = evaluator.evaluate()
        return result

    def optimizeTrackerConfig(self, fileName, trackerTypes, ctRange: Tuple[float, float], rtRange: Tuple[float, float], stRange: Tuple[float, float], smpRange: Tuple[float, float], mspRange: Tuple[float, float], videosList: Sequence[Tuple[str, str, str]] = None, noIterations: int = None, noFrames: int=30, withPartialOutput: bool = False, stepSize: float = None, maxAgeRange: Sequence[int] = None):
        if videosList is None:
            videosList = self.getTuningVideoList()
        print("Working on ", videosList)
        baseDir = 'results'
        keys = ["trackerType", "createThreshold", "removeThreshold", "surviveThreshold", "surviveMovePercent", "minScorePrediction", "maxAge"]
        evaluations = ['mAP', "Elapsed time", "GTmAP", "Memory"]
        titles = keys + evaluations
        # keys = keys + evaluations
        emptyDirectory(baseDir)
        results = HyperParametersTuner.tryToFindBestConfig(self.config, videosList,
                                                           noIterations, trackerTypes,
                                                           ctRange, rtRange, stRange, smpRange, mspRange,
                                                           noFrames, withPartialOutput, stepSize, maxAgeRange)
        f = open(os.path.join(baseDir, fileName), "w")
        stringPattern = '{:18s}'
        floatPattern = '{:18f}'
        bar = " | "
        f.write((stringPattern + (bar + stringPattern)*(len(titles) - 1) + "\n").format(*titles))
        answer = []
        for result in results:
            params = ([result[0].getTrackingHyperParameters()[x] for x in keys] + [result[1][metric] for metric in evaluations])
            answer.append(params)
            f.write((stringPattern + (bar + floatPattern)*(len(titles) - 1) + "\n").format(*params))
        f.close()
        return titles, answer



    def playVideo(self, video: Tuple[str, str, str], config: BasicConfig = None, nrFrames = MAX_VIDEO_LENGTH) -> None:
        if config is None:
            config = self.config
        gtPredictor = GroundTruthPredictor(video[0], video[1], video[2])
        yoloPredictor = YOLOPredictor(gtPredictor, config)
        trackerPredictor = TrackerPredictor(yoloPredictor, gtPredictor, getTrackerFromConfig(config), config)
        trackerPredictor = MinScoreWrapperPredictor(trackerPredictor, config.minScorePrediction)
        RED = [0, 0, 255]
        GREEN = [0, 255, 0]
        BLUE = [255, 0, 0]
        print(config)
        playVideo([(yoloPredictor, RED), (gtPredictor, GREEN), (trackerPredictor, BLUE)],
                  gtPredictor, nrFrames)

    def generateNewData(self, config: BasicConfig, videoList: Sequence[Tuple[str, str, str]] = None, verbose: bool = False, nrFrames: int = MAX_VIDEO_LENGTH) -> None:
        if videoList is None:
            videoList = self.getGenerationVideoList()
        NewDataGenerator.initializeDirectory(IMAGE_GENERATION_SAVE_PATH)
        # gtPredictors = findGroundTruthFromVideoList(videoList)
        gtPredictors = []
        tracker = getTrackerFromConfig(config)
        # i = 0
        toIterate = videoList
        if verbose:
            toIterate = tqdm(toIterate)
        for datasetName, setName, videoName in toIterate:
            gtPredictor = GroundTruthPredictor(datasetName, setName, videoName)
            gtPredictors.append(gtPredictor)
            yoloPredictor = YOLOPredictor(gtPredictor, config)
            trackerPredictor = TrackerPredictor(yoloPredictor, gtPredictor, tracker, config)
            actualPredictor = MinScoreWrapperPredictor(trackerPredictor, config.minScorePrediction)
            generator = NewDataGenerator(actualPredictor, gtPredictor, self.imgSaveTextPattern)
            generator.generateNewData(config.imageGenerationSavePeriod, IMAGE_GENERATION_SAVE_PATH,
                                      config.imageGenerationSaveFileName, verbose, nrFrames)

    def retrain(self, config: BasicConfig) -> None:
        newAnnotationsFile = os.path.join(IMAGE_GENERATION_SAVE_PATH, config.imageGenerationSaveFileName)
        trainer = YOLOTrainer(config, [ANNOTATIONS_FILE, newAnnotationsFile])
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

    def getAllTrainIds(self):
        path = os.path.join(MODELS_DIR)
        trainIdsList = os.listdir(path)
        rez = []
        for trainId in trainIdsList:
            try:
                x = int(trainId)
                rez.append(x)
            except ValueError:
                pass

        return sorted(rez)

    @staticmethod
    def createNewTrainingConfiguration(trainId):
        config = BasicConfig()
        config.trainId = trainId
        saveConfiguration(config)

    @staticmethod
    def getAllAvailableTrackerTypes():
        return ["fake", "csrt", "kcf", "boosting", "mil", "tld", "medianflow", "mosse"]


