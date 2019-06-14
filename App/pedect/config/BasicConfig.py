import pickle

from pedect.utils.constants import *


class BasicConfig:
    def save(self, configFile: str = None):
        if configFile is None:
            configFile = os.path.join(MODELS_DIR, self.trainId, "config.pickle")
        self.updateDictionary()
        f = open(configFile, 'wb+')
        pickle.dump(self, f)
        f.close()

    def saveText(self, configFile: str = None):
        if configFile is None:
            configFile = os.path.join(MODELS_DIR, self.trainId, "config.txt")
        f = open(configFile, 'w+')
        f.write(str(self))
        f.close()

    def getDictionary(self):
        dict1 = BasicConfig.__dict__
        dict = {}
        for k, v in dict1.items():
            if not callable(v) and k[0] != '_':
                dict[k] = getattr(self, k)
        return dict

    def updateDictionary(self):
        for k, v in self.getDictionary().items():
            self.__dict__[k] = v

    # possibleLabels = {'people': (255, 0, 0), 'person-fa': (0, 0, 255), 'person': (0, 255, 0)}
    # For training
    trainId = "2"
    modelName = "trained_weights_final.h5"
    inputShape = (416, 416)  # multiple of 32, hw
    freezeNoEpochs = 10
    noFreezeNoEpochs = 10
    isTiny = False
    validationSplit = 0.1
    freezeBatchSize = 16
    noFreezeBatchSize = 16
    loadPreTrained = True
    preTrainedModelPath = "default"

    def getPreTrainedModelPath(self):
        sign = '/'
        if sign not in self.preTrainedModelPath:
            sign = '\\'
        words = self.preTrainedModelPath.split(sign)
        return os.path.join(*words)

    checkpointPeriod = 150
    initialLR = 1e-4
    alreadyTrainedEpochs = 0

    # For tracking
    createThreshold = 0.9
    removeThreshold = 0.5
    surviveThreshold = 0.2
    surviveMovePercent = 0.0
    minScorePrediction = 0.5
    maxNrOfObjectsPerFrame = 10
    maxAge = 100

    imageGenerationSavePeriod = 10
    imageGenerationSaveFileName = "annotations.csv"

    batchSplit = (0.2, 0.1, 0.05, 0.65)

    trackerType = "medianflow"

    def getModelPath(self):
        return os.path.join(MODELS_DIR, str(self.trainId), self.modelName)

    def getAnchorsPath(self):
        if self.isTiny:
            return os.path.join(YOLO_DIR, 'model_data', 'tiny_yolo_anchors.txt')
        return os.path.join(YOLO_DIR, 'model_data', 'yolo_anchors.txt')

    def getPredictionsPath(self):
        return os.path.join(PREDICTIONS_PATH, str(self.trainId) + "-" + str(self.alreadyTrainedEpochs))

    @staticmethod
    def configName():
        return "BasicConfig"

    def getTrackingHyperParameters(self):
        return {
            "trackerType": self.trackerType,
            "createThreshold": self.createThreshold,
            "removeThreshold": self.removeThreshold,
            "surviveThreshold": self.surviveThreshold,
            "surviveMovePercent": self.surviveMovePercent,
            "minScorePrediction": self.minScorePrediction,
            "maxAge": self.maxAge,
            "maxNrOfObjectsPerFrame": self.maxNrOfObjectsPerFrame
        }

    def __str__(self):
        res = ""
        for k, v in self.getDictionary().items():
            res += "%s = %s\n" % (k, str(v))
        return res

def saveConfiguration(config: BasicConfig):
    models_dir = os.path.join(MODELS_DIR, config.trainId)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    config.save()
    config.saveText()

def getConfig(fileName = "") -> BasicConfig:
    try:
        f = open(fileName, "rb")
        result = pickle.load(f)
        f.close()
        return result
    except IOError:
        return None

def getConfigFromTrainId(trainId) -> BasicConfig:
    return getConfig(os.path.join(MODELS_DIR, str(trainId), "config.pickle"))

def getSavedConfigIfExists(config: BasicConfig):
    saved = getConfigFromTrainId(config.trainId)
    if saved is not None:
        return saved
    return config
