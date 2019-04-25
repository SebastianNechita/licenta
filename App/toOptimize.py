from pedect.config.BasicConfig import *
import sys
from pedect.controller.Controller import Controller
from pedect.tracker.OpenCVTracker import OpenCVTracker
from pedect.evaluator.HyperParametersTuner import *


sys.path.append("./keras-yolo3/") 
sys.path.append("./re3-tensorflow/") 
print(sys.path)

class MyConfig(BasicConfig):
    trainId = "5"
    inputShape = (int(480 / 32 // 2 * 32), int(640 / 32 // 2 * 32))
    noFreezeNoEpochs = 5
    noFreezeBatchSize = 32
    loadPretrained = False
    checkpointPeriod = 3
MyConfig()
tracker = OpenCVTracker("medianflow")
config = getConfigFromTrainId(2)
controller = Controller(config, tracker)


ctRange = (0.0, 1.0)
rtRange = (0.0, 1.0)
stRange = (0.0, 1.0)
smpRange = (0.0, 1.0)
mspRange = (0.0, 1.0)

# videosList = [ ("caltech", "set03", "V000"), ("caltech" , "set01", "V005"), ("caltech", "set01", "V000")]
videosList = [("caltech" , "set01", "V005")]
controller.optimizeTrackerConfig(ctRange, rtRange, stRange, smpRange, mspRange, videosList, 100, 1000, False)
