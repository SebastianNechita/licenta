from pedect.converter.ConverterToImages import *
from pedect.config.BasicConfig import *
import os
import sys

sys.path.append("./keras-yolo3/") 
sys.path.append("./re3-tensorflow/") 
print(sys.path)

from pedect.controller.Controller import Controller

class MyConfig(BasicConfig):
    trainId = "4"
    inputShape = (int(480 / 32 // 2 * 32), int(640 / 32 // 2 * 32))
    noFreezeNoEpochs = 5
    noFreezeBatchSize = 32
    loadPretrained = False
    checkpointPeriod = 3
config = MyConfig()

controller = Controller(config)

from pedect.controller.Controller import Controller
   

config = getConfigFromTrainId(5)
controller = Controller(config)

config.batchSplit = (0.2, 0.1, 0.04, 0.66)
len(controller.splitIntoBatches()[2])

from pedect.evaluator.HyperParametersTuner import *

ctRange = (0.0, 1.0)
rtRange = (0.0, 1.0)
stRange = (0.0, 1.0)
smpRange = (0.0, 1.0)
mspRange = (0.0, 1.0)

controller.optimizeTrackerConfig(ctRange, rtRange, stRange, smpRange, mspRange, None, 30, 10000, True)