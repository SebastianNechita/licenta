from pedect.config.BasicConfig import *
import sys
from pedect.service.Service import Service
from pedect.evaluator.HyperParametersTuner import *
import random

from pedect.tracker.TimesHolder import TimesHolder

random.seed(1)

sys.path.append("./keras-yolo3/") 
sys.path.append("./re3-tensorflow/") 

class MyConfig(BasicConfig):
    pass

MyConfig()
config = getConfigFromTrainId(11)
# config.trackerType = "csrt"
# config.trackerType = "fake"
# config.trackerType = "kcf"
# config.trackerType = "cached medianflow"
config.trackerType = "medianflow"
# config.trackerType = "cached csrt"
# config.trackerType = "cached boosting"
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
videosList = [("caltech", "set01", "V004")]
# videosList = [("caltech", "set03", "V009")]
# videosList = None
rangeSize = 2

service.optimizeTrackerConfig(ctRange, rtRange, stRange, smpRange, mspRange, videosList, rangeSize ** howMany, 100, True, rangeSize)
print("Time slices are:")
print([("%.2f%%" % x) for x in service.getRunningTimesPercentForTracker()])
print("Tracker calls / CachedTracker calls / Percent")
ta = TimesHolder.trackerAccessed
cta = TimesHolder.cachedTrackerAccessed
print(ta, cta)
print(ta, cta, "%.2f%%" % ((ta * 100) / max(ta, cta)))