from keras_retinanet.bin.debug import *

from pedect.utils.constants import *

def debug(config):
    options = {
        "--image-max-side": config.imageMaxSide,
    }
    optsList = []
    for key in options.keys():
        optsList.append(key)
        optsList.append(str(options[key]))
    optsList.append("csv")
    optsList.append(ANNOTATIONS_FILE)
    optsList.append(LABELS_FILE)
    main(args=optsList)