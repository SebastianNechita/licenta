from typing import Sequence

import numpy as np
from PIL import Image
from yolo import YOLO
from yolo3.utils import letterbox_image
from keras import backend as K

from pedect.config.BasicConfig import BasicConfig
from pedect.predictor.Predictor import Predictor
from pedect.predictor.VideoHolder import VideoHolder
from pedect.utils.constants import *

from pedect.predictor.PredictedBox import PredictedBox
from pedect.utils.osUtils import *

class YOLOManager:
    existentYoloObjects = {}
    __cache = {}

    @staticmethod
    def getYoloObject(config: BasicConfig):
        uniqueValue = (config.getModelPath(), config.getAnchorsPath())
        if uniqueValue not in YOLOManager.existentYoloObjects:
            print("Created a new YOLO object!")
            K.clear_session()
            YOLOManager.existentYoloObjects.clear()
            YOLOManager.existentYoloObjects[uniqueValue] = YOLO(model_path = config.getModelPath(),
                                                                classes_path = LABELS_FILE,
                                                                anchors_path = config.getAnchorsPath(),
                                                                score = 0.05)
        return YOLOManager.existentYoloObjects[uniqueValue]

    @staticmethod
    def getGlobalCache():
        return YOLOManager.__cache

class YOLOPredictor(Predictor):
    # def finishPrediction(self):
    #     pass

    def __init__(self, videoHolder: VideoHolder, config: BasicConfig):
        self.videoHolder = videoHolder
        self.config = config
        self.yoloObject = None
        self.savePath = os.path.join(self.config.getPredictionsPath(), self.videoHolder.chosenDataset.datasetName,
                                     self.videoHolder.setName, self.videoHolder.videoNr)
        self.cache = YOLOManager.getGlobalCache() if USE_GLOBAL_PREDICTION_CACHE else {}

    def getPredictionPathForFrame(self, frameNr: int):
        return os.path.join(self.savePath, str(frameNr) + ".prediction")

    def readPredictionBoxes(self, predictionPath: str):
        noTries = 10
        for i in range(noTries):
            try:
                f = open(predictionPath, "r")
                boxes = []
                for line in f.readlines():
                    v = line.split("\n")[0].split(" ")
                    if len(v) < 6:
                        continue
                    boxes.append(PredictedBox(int(v[0]), int(v[1]), int(v[2]), int(v[3]), v[5], float(v[4])))
                f.close()
                return boxes
            except IOError as e:
                if i < noTries - 1:
                    print("Small input error %d / %d. Trying again!" % (i, noTries))
                raise e

    def writePredictionBoxes(self, predictionPath: str, objects: Sequence[PredictedBox]):
        createDirectoryIfNotExists(self.savePath)
        f = open(predictionPath, "w+")
        for obj in objects:
            f.write("%d %d %d %d %f %s\n" % (obj.getX1(),
                                             obj.getY1(),
                                             obj.getX2(),
                                             obj.getY2(),
                                             obj.getProb(),
                                             obj.getLabel()))
        f.close()

    def predictForFrame(self, frameNr: int):
        predictionPath = self.getPredictionPathForFrame(frameNr)
        if predictionPath in self.cache:
            return self.cache[predictionPath]
        if os.path.isfile(predictionPath):
            objects = self.readPredictionBoxes(predictionPath)
            self.cache[predictionPath] = objects
            return objects
        if self.yoloObject is None:
            self.yoloObject = YOLOManager.getYoloObject(self.config)
        image = Image.fromarray(self.videoHolder.getFrame(frameNr), 'RGB')
        if self.yoloObject.model_image_size != (None, None):
            assert self.yoloObject.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.yoloObject.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.yoloObject.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.yoloObject.sess.run(
            [self.yoloObject.boxes, self.yoloObject.scores, self.yoloObject.classes],
            feed_dict={
                self.yoloObject.yolo_model.input: image_data,
                self.yoloObject.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        objects = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.yoloObject.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int'))
            left = max(0, np.floor(left + 0.5).astype('int'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int'))
            objects.append(PredictedBox(int(left), int(top), int(right), int(bottom), predicted_class, float(score)))

        self.writePredictionBoxes(predictionPath, objects)
        self.cache[predictionPath] = objects
        return objects
