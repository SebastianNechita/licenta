from pedect.input.inputProcessing import *
from pedect.predictor.Predictor import Predictor
from pedect.predictor.PredictedBox import PredictedBox
from pedect.predictor.VideoHolder import *


class GroundTruthPredictor(Predictor, VideoHolder):
    def __init__(self, chosenDataset, setName: str, videoNr: str):
        if isinstance(chosenDataset, str):
            chosenDataset = findDatasetByName(chosenDataset)
        VideoHolder.__init__(self, chosenDataset, setName, videoNr)
        annotations_path = chosenDataset.getAnnotationsPath(setName, videoNr)
        self.annotations = read_vbb(annotations_path)

    def predictForFrame(self, frameNr: int):
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


