from keras_retinanet.bin.evaluate import *

from pedect.utils.constants import *


def evaluate(config):
    options = {
    }
    optsList = []
    for key in options.keys():
        optsList.append(key)
        optsList.append(str(options[key]))
    optsList.append("csv")
    # optsList.append('-h')
    modelPath = os.path.join("snapshots", "inference", "mobilenet128_1.0_csv_100.h5")
    optsList.append(ANNOTATIONS_FILE)
    optsList.append(LABELS_FILE)
    optsList.append(modelPath)

    print(optsList)
    main(args = optsList)