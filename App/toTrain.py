# import tensorflow as tf
# print(tf.__file__)

# tf.reset_default_graph()

from pedect.config.BasicConfig import *
import sys
from pedect.service.Service import Service
from pedect.tracker.OpenCVTracker import OpenCVTracker
from pedect.evaluator.HyperParametersTuner import *
import random
random.seed(1)

sys.path.append("./keras-yolo3/")
sys.path.append("./re3-tensorflow/")

divisionRate = 2

class MyConfig(BasicConfig):
    trainId = "11"
    inputShape = (int(480 / 32 // divisionRate * 32), int(640 / 32 // divisionRate * 32))
    noFreezeNoEpochs = 50
    freezeNoEpochs = 50
    freezeBatchSize = 32
    noFreezeBatchSize = 32
    loadPretrained = True
    checkpointPeriod = 5
    trackerType = "fake"
    isTiny = True

config = MyConfig()
print("Input shape: ", config.inputShape)
tracker = OpenCVTracker(config.trackerType)
service = Service(config, tracker)

# service.prepareTrainingSet()
config.initialLR = 1e-4
service.train()

# divRate = 1, isTiny = True -> batchSize = 32 ---- 2min
# divRate = 2, isTiny = True -> batchSize = 128 ---- 41 sec (35)
# divRate = 2, isTiny = False -> batchSize = 32 ---- 73 sec (60)
# divRate = 2, isTiny = False -> batchSize = 16 ---- 85 sec (73)
# winner = divRate = 2, isTiny = True


