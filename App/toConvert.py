import sys

from pedect.service.Service import Service

sys.path.append("./keras-yolo3/")
sys.path.append("./re3-tensorflow/")

service = Service()
service.prepareTrainingSet()

