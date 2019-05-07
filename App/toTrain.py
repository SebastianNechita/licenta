from pedect.config.BasicConfig import *
import sys
from pedect.service.Service import Service
from pedect.tracker.OpenCVTracker import OpenCVTracker
from pedect.evaluator.HyperParametersTuner import *
import random
random.seed(1)

sys.path.append("./keras-yolo3/")
sys.path.append("./re3-tensorflow/")

divisionRate = 4

class MyConfig(BasicConfig):
    trainId = "6"
    inputShape = (int(480 / 32 // divisionRate * 32), int(640 / 32 // divisionRate * 32))
    noFreezeNoEpochs = 100
    noFreezeBatchSize = 64
    loadPretrained = False
    checkpointPeriod = 3
    trackerType = "fake"
    isTiny = True

config = MyConfig()
print("Input shape: ", config.inputShape)
tracker = OpenCVTracker(config.trackerType)
service = Service(config, tracker)

# service.prepareTrainingSet([("caltech", "set01", "V004")])
config.initialLR = 1e-3
config.LRDecayPeriod = 10
config.LRDecayMagnitude = 0.8
service.train()