from keras_retinanet.bin.train import *

from pedect.utils.constants import *


def train(config):
    options = {
        "--image-max-side": config.imageMaxSide,
        "--batch-size": config.batchSize,
        "--epochs": config.nrEpochs,
        # "--backbone": "mobilenet128_1.0",
        "--steps": config.steps
    }
    optsList = []
    for key in options.keys():
        optsList.append(key)
        optsList.append(str(options[key]))
    optsList.append("csv")
    optsList.append(ANNOTATIONS_FILE)
    optsList.append(LABELS_FILE)

    main(args = optsList)