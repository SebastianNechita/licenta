import copy
import os
import pickle

from pedect.utils.constants import MODELS_DIR, YOLO_DIR


class BasicConfig:
    def save(self, configFile):
        f = open(configFile, 'wb')
        pickle.dump(self, f)
        f.close()

    def saveText(self, configFile):
        f = open(configFile, 'w')
        f.write(str(self))
        f.close()

    def getDictionary(self):
        dict1 = BasicConfig.__dict__
        dict2 = BasicConfig.__bases__[0].__dict__
        dict = {}
        for k, v in dict1.items():
            if k in self.__dict__:
                v = self.__dict__[k]
            if k not in dict2 and not callable(v) and k[0] != '_':
                dict[k] = v
        return dict

    possibleLabels = {'people': (255, 0, 0), 'person-fa': (0, 0, 255), 'person': (0, 255, 0)}
    # For training
    trainId = "1"
    modelName = "trained_weights_final.h5"
    inputShape = (416, 416)  # multiple of 32, hw
    freezeNoEpochs = 1
    noFreezeNoEpochs = 0
    isTiny = True
    validationSplit = 0.3
    freezeBatchSize = 5
    noFreezeBatchSize = 1
    loadPretrained = True
    # For tracking
    createThreshold = 0.9
    removeThreshold = 0.5
    surviveThreshold = 0.2
    surviveMovePercent = 0.0
    maxAge = 100
    checkpointPeriod = 1

    def getModelPath(self):
        return os.path.join(MODELS_DIR, str(self.trainId), self.modelName)

    def getAnchorsPath(self):
        if self.isTiny:
            return os.path.join(YOLO_DIR, 'model_data', 'tiny_yolo_anchors.txt')
        return os.path.join(YOLO_DIR, 'model_data', 'yolo_anchors.txt')

    def getPredictionsPath(self):
        return os.path.join(PREDICTIONS_PATH, str(self.trainId))

    def configName(self):
        return "BasicConfig"

    def __str__(self):
        res = ""
        for k, v in self.getDictionary().items():
            res += "%s = %s\n" % (k, str(v))
        return res


def getConfig(fileName = ""):
    if fileName == "":
        return BasicConfig()
    f = open(fileName, "rb")
    result = pickle.load(f)
    f.close()
    return result

def getConfigFromTrainId(trainId):
    return getConfig(os.path.join(MODELS_DIR, str(trainId), "config.pickle"))
