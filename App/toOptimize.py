from pedect.config.BasicConfig import *
import sys

from pedect.service.Service import Service
from pedect.evaluator.HyperParametersTuner import *
import random

from pedect.tracker.TimesHolder import TimesHolder
from pedect.utils.osUtils import emptyDirectory

random.seed(1)

sys.path.append("./keras-yolo3/") 
sys.path.append("./re3-tensorflow/") 

class MyConfig(BasicConfig):
    pass

MyConfig()
config = getConfigFromTrainId(11)
config.trackerType = "kcf"
print("Tracker type is", config.trackerType)
service = Service()
emptyDirectory(TEMP_FOLDER)
trackerTypes = ["kcf", "fake"]
ctRange = (0.0, 0.0)
rtRange = (0.45, 0.45)
stRange = (0.0, 0.0)
smpRange = (0.0, 0.0)
mspRange = (0.0, 0.0)
ranges = [ctRange, rtRange, stRange, smpRange, mspRange]
maxAgeRange = [2000]
maxObjectsRange = [50]
# maxAgeRange = None
# videosList = [ ("caltech", "set03", "V000"), ("caltech" , "set01", "V005"), ("caltech", "set01", "V000")]
# videosList = [("caltech", "set01", "V004")]
# videosList = [("caltech", "set03", "V009")]
videosList = None
# videosList = service.getTrainingVideoList()
stepSize = 0.01
# service.playVideo(("caltech", "set03", "V009"), config)
service.optimizeTrackerConfig(config, "a.txt", trackerTypes, ctRange, rtRange, stRange, smpRange, mspRange, videosList, None, MAX_VIDEO_LENGTH, True, stepSize, maxAgeRange, maxObjectsRange)
print("Time slices are:")
print([("%.2f%%" % x) for x in service.getRunningTimesPercentForTracker()])
print("Tracker calls / CachedTracker calls / Percent")
ta = TimesHolder.trackerAccessed
cta = TimesHolder.cachedTrackerAccessed
print(ta, cta)
print(ta, cta, "%.2f%%" % ((ta * 100) / max(ta, cta)))
