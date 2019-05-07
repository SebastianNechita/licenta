from pedect.converter.ConverterToImages import *
from pedect.config.BasicConfig import *
import os
import sys

sys.path.append("./keras-yolo3/") 
sys.path.append("./re3-tensorflow/") 
print(sys.path)

from pedect.service.Service import Service

class MyConfig(BasicConfig):
    trainId = "5"
    inputShape = (int(480 / 32 // 2 * 32), int(640 / 32 // 2 * 32))
    noFreezeNoEpochs = 30
    noFreezeBatchSize = 16
    loadPretrained = False
    checkpointPeriod = 3
config = MyConfig()

controller = Service(config)

# service.prepareTrainingSet()
controller.train()
