from keras_retinanet.models import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import cv2
import os
import time
import matplotlib.pyplot as plt
from keras_retinanet.utils.visualization import *


def evaluate(modelPath, config):
    inferenceModel = load_model(modelPath, backbone_name=config.backbone)

    for i in range(1169, 1183):

        labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car'}
        image = read_image_bgr(os.path.join("..", "Data", "images", "set00-V000-" + str(i) + ".jpg"))
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = inferenceModel.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)
        print(sum(sum(boxes)))
        # correct for image scale
        boxes /= scale
        print(scores[0])
        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            print(score)
            if score < 0.5:
                break

            color = label_color(label)

            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)

        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()