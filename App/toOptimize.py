from pedect.config.BasicConfig import *
import sys
from pedect.service.Service import Service
from pedect.tracker.OpenCVTracker import OpenCVTracker
from pedect.evaluator.HyperParametersTuner import *
import random
random.seed(1)

sys.path.append("./keras-yolo3/") 
sys.path.append("./re3-tensorflow/") 

class MyConfig(BasicConfig):
    trainId = "5"
    inputShape = (int(480 / 32 // 2 * 32), int(640 / 32 // 2 * 32))
    noFreezeNoEpochs = 5
    noFreezeBatchSize = 32
    loadPretrained = False
    checkpointPeriod = 3

MyConfig()
config = getConfigFromTrainId(5)
config.trackerType = "csrt"
config.trackerType = "fake"
# config.trackerType = "medianflow"
print("Tracker type is", config.trackerType)
tracker = OpenCVTracker(config.trackerType)
controller = Service(config, tracker)


ctRange = (0.0, 0.0)
rtRange = (0.0, 1.0)
stRange = (0.0, 0.0)
smpRange = (0.0, 0.0)
mspRange = (0.0, 0.0)
ranges = [ctRange, rtRange, stRange, smpRange, mspRange]
howMany = 0
for i in ranges:
    if i[0] != i[1]:
        howMany = howMany + 1
# videosList = [ ("caltech", "set03", "V000"), ("caltech" , "set01", "V005"), ("caltech", "set01", "V000")]
videosList = [("caltech", "set01", "V004")]
# videosList = None
rangeSize = 11
howMany = 1
controller.optimizeTrackerConfig(ctRange, rtRange, stRange, smpRange, mspRange, videosList, rangeSize ** howMany, 50, True, rangeSize)
