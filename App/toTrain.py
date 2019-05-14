import sys

from pedect.config.BasicConfig import getSavedConfigIfExists
from pedect.service.Service import Service
from pedect.evaluator.HyperParametersTuner import *
import random
random.seed(1)

sys.path.append("./keras-yolo3/")
sys.path.append("./re3-tensorflow/")

divisionRate = 4

class MyConfig(BasicConfig):
    trainId = "12"
    inputShape = (int(480 / 32 // divisionRate * 32), int(640 / 32 // divisionRate * 32))
    freezeNoEpochs = 1
    noFreezeNoEpochs = 1
    freezeBatchSize = 32
    noFreezeBatchSize = 32
    loadPreTrained = True
    checkpointPeriod = 5
    isTiny = True

config = getSavedConfigIfExists(MyConfig())
config.freezeNoEpochs = 0
config.noFreezeNoEpochs = 3

print("Input shape: ", config.inputShape)
service = Service(config)

# service.prepareTrainingSet()
config.initialLR = 1e-4
service.train()

# divRate = 1, isTiny = True -> batchSize = 32 ---- 2min
# divRate = 2, isTiny = True -> batchSize = 128 ---- 41 sec (35)
# divRate = 2, isTiny = False -> batchSize = 32 ---- 73 sec (60)
# divRate = 2, isTiny = False -> batchSize = 16 ---- 85 sec (73)
# winner = divRate = 2, isTiny = True


