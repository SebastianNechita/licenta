from keras_retinanet.bin.convert_model import *
import os


name = "mobilenet128_1.0_csv_100.h5"
def convertModel(name, config):
    baseDir = "snapshots"
    infDir = os.path.join(baseDir, "inference")
    path1 = os.path.join(baseDir, name)
    path2 = os.path.join(infDir, name)
    if not os.path.exists(infDir):
        os.mkdir(infDir)
    options = {
        "--backbone": config.backbone,
    }
    optsList = []
    for key in options.keys():
        optsList.append(key)
        optsList.append(str(options[key]))
    optsList.append(path1)
    optsList.append(path2)
    main(args=optsList)