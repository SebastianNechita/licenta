from pedect.config.BasicConfig import *
import sys

from pedect.service.Service import Service
from pedect.evaluator.HyperParametersTuner import *
import random

from pedect.utils.osUtils import emptyDirectory

random.seed(1)

sys.path.append("./keras-yolo3/")
sys.path.append("./re3-tensorflow/")

class MyConfig(BasicConfig):
    pass

MyConfig()
config = getConfigFromTrainId(11)
print(config)
service = Service()
emptyDirectory(TEMP_FOLDER)
trackerTypes = ["kcf", "fake"]

# videosList = [ ("caltech", "set03", "V000"), ("caltech" , "set01", "V005"), ("caltech", "set01", "V000")]
# videosList = [("caltech", "set01", "V004")]
videosList = [("caltech", "set03", "V009")]
# videosList = None
# videosList = service.getTrainingVideoList()

service.generateNewData(config, videosList, True)
