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
    pass

MyConfig()
config = getConfigFromTrainId(11)
config.trackerType = "csrt"
config.trackerType = "fake"
config.trackerType = "kcf"
config.trackerType = "medianflow"
config.trackerType = "boosting"
# config.trackerType = "mil"
# config.trackerType = "mosse"
print("Tracker type is", config.trackerType)
service = Service(config)


ctRange = (0.0, 1.0)
rtRange = (0.0, 1.0)
stRange = (0.0, 1.0)
smpRange = (0.0, 1.0)
mspRange = (0.0, 1.0)
ranges = [ctRange, rtRange, stRange, smpRange, mspRange]
howMany = 0
for i in ranges:
    if i[0] != i[1]:
        howMany = howMany + 1
# videosList = [ ("caltech", "set03", "V000"), ("caltech" , "set01", "V005"), ("caltech", "set01", "V000")]
# videosList = [("caltech", "set01", "V004")]
videosList = None
rangeSize = 3

service.optimizeTrackerConfig(ctRange, rtRange, stRange, smpRange, mspRange, videosList, rangeSize ** howMany, 300, True, rangeSize)
print([("%.2f%%" % x) for x in service.getRunningTimesPercentForTracker()])