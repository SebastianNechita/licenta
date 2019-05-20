import os

from PIL import Image
from tqdm import tqdm

from pedect.input.labelsHelper import getAllPossibleLabelsDictionary
from pedect.predictor.Predictor import Predictor
from pedect.predictor.VideoHolder import VideoHolder
from pedect.utils.constants import MAX_VIDEO_LENGTH
from pedect.utils.osUtils import createDirectoryIfNotExists, emptyDirectory


class NewDataGenerator:
    def __init__(self, predictor: Predictor, videoHolder: VideoHolder, textPattern: str):
        self.predictor = predictor
        self.videoHolder = videoHolder
        self.textPattern = textPattern

    def generateNewData(self, selectPeriod: int, saveFolder: str, saveFileName: str, verbose: bool = False, frameNr: int = MAX_VIDEO_LENGTH) -> None:
        toIterate = range(min(self.videoHolder.getLength(), frameNr))
        if verbose:
            toIterate = tqdm(toIterate)
        predictions = []
        for i in toIterate:
            predictions.append((self.videoHolder.getFrame(i), self.predictor.predictForFrame(i)))
        # predictions = [(self.videoHolder.getFrame(i), self.predictor.predictForFrame(i))
        #                for i in toIterate]
        predictions = predictions
        saveFilePath = os.path.join(saveFolder, saveFileName)
        f = open(saveFilePath, 'a+')
        labelsDictionary = getAllPossibleLabelsDictionary()
        for i in range(len(predictions))[::selectPeriod]:
            img, pred = predictions[i]
            imgName = self.textPattern % (self.videoHolder.chosenDataset.datasetName, self.videoHolder.setName,
                                          self.videoHolder.videoNr, i)
            imgPath = os.path.join(saveFolder, imgName)
            Image.fromarray(img).save(imgPath)
            string = imgPath
            for prediction in pred:
                string += " %d,%d,%d,%d,%d" % (prediction.getX1(), prediction.getY1(),
                                               prediction.getX2(), prediction.getY2(),
                                               labelsDictionary[prediction.getLabel()])
            string += "\n"
            f.write(string)
        f.close()

    @staticmethod
    def initializeDirectory(saveFolder: str) -> None:
        emptyDirectory(saveFolder)
        createDirectoryIfNotExists(saveFolder)

