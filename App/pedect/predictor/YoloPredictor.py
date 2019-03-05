import numpy as np
from PIL import Image
from yolo import YOLO
from yolo3.utils import letterbox_image
from keras import backend as K

from pedect.predictor.BasePredictor import BasePredictor
from pedect.utils.constants import *

from pedect.predictor.PredictedBox import PredictedBox

class YoloPredictor(BasePredictor):
    def __init__(self, videoHolder, config):
        self.videoHolder = videoHolder
        self.yoloObject = YOLO(model_path = config.modelPath, classes_path = LABELS_FILE, anchors_path = config.anchorsPath)

    def predictForFrame(self, frameNr):
        image = Image.fromarray(self.videoHolder.getFrame(frameNr), 'RGB')
        if self.yoloObject.model_image_size != (None, None):
            assert self.yoloObject.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.yoloObject.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.yoloObject.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
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
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            objects.append(PredictedBox(top, left, bottom, right, predicted_class, score))
        return objects