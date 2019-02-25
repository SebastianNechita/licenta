from pedect.input.inputProcessing import *
from pedect.predictor.PredictedBox import PredictedBox


class BasePredictor:
    def __init__(self, chosenDataset, setName, videoNr):
        video_path = chosenDataset.getVideoPath(setName, videoNr)
        annotations_path = chosenDataset.getAnnotationsPath(setName, videoNr)
        self.video = read_seq(video_path)
        self.annotations = read_vbb(annotations_path)

    def getLength(self):
        return len(self.video)

    def getVideo(self):
        return self.video

    def getFrame(self, frameNr):
        return self.video[frameNr]

    def predictForFrame(self, frameNr):
        raise Exception("Not implemented!")

    def getGroundTruthForFrame(self, frameNr):
        anns = getAnnotationsForFrame(frameNr, self.annotations)
        predictedBBoxes = []
        for ann in anns:
            x1 = int(ann["pos"][0])
            y1 = int(ann["pos"][1])
            x2 = int(ann["pos"][2]) + x1
            y2 = int(ann["pos"][3]) + y1
            if x1 >= x2 or y1 >= y2:
                continue
            label = ann["lbl"]
            if label == 'person?':
                continue
            predictedBBoxes.append(PredictedBox(x1, y1, x2, y2, label, 1.0))
        return predictedBBoxes
