import os

from pedect.utils.osUtils import createDirectoryIfNotExists


class NewDataGenerator:
    def __init__(self, predictor, videoHolder, textPattern):
        self.predictor = predictor
        self.videoHolder = videoHolder
        self.textPattern = textPattern

    def generateNewData(self, selectPeriod, saveFolder, saveFileName):
        predictions = [(self.videoHolder.getFrame(i), self.predictor.predictForFrame(i))
                       for i in range(self.videoHolder.getLength())]
        predictions = predictions
        createDirectoryIfNotExists(saveFolder)
        saveFilePath = os.path.join(saveFolder, saveFileName)
        f = open(saveFilePath, 'a+')
        for i in range(len(predictions))[::selectPeriod]:
            img, pred = predictions[i]
            imgName = self.textPattern % (self.videoHolder.chosenDataset.datasetName, self.videoHolder.setName,
                                          self.videoHolder.videoNr, i)
            imgPath = os.path.join(saveFolder, imgName)
            img.save(imgPath)
            string = imgPath
            for prediction in pred:
                string += " %d,%d,%d,%d,%d" % (prediction.getX1(), prediction.getY1(),
                                               prediction.getX2(), prediction.getY2(), prediction.getLabel())
            imgPath += "\n"
            f.write(string)
        f.close()
