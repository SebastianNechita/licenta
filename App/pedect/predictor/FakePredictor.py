import random

from pedect.predictor.Predictor import Predictor


class FakePredictor(Predictor):
    # fakeProbRange = (0.0, 0.8)
    # realProbRange = (0.3, 1.0)

    def __init__(self, fakeProbRange, realProbRange, predictor, videoHolder):
        self.fakeProbRange = fakeProbRange
        self.realProbRange = realProbRange
        self.predictor = predictor
        self.videoHolder = videoHolder

    def getPredictedBoxesForFrame(self, frameNr):
        boxes = self.predictor.predictForFrame(frameNr)
        for box in boxes:
            box.setProb(random.random() * (self.realProbRange[1] - self.realProbRange[0]) + self.realProbRange[0])
        return boxes

    @staticmethod
    def __limitBetween(number, left, right):
        if number < left:
            number = left
        if number > right:
            number = right
        return number

    def getFakePredictionsForFrame(self, frameNr):
        imageShape = self.videoHolder.getFrame(0).shape
        predictedBBoxes = []
        correctPredictions = self.getPredictedBoxesForFrame(frameNr) + self.getPredictedBoxesForFrame(frameNr)
        for pred in correctPredictions:
            value = 100
            pred.setX1(self.__limitBetween(pred.getX1() + int((random.random() - 0.5) * 2 * value), 0, imageShape[1] - 1))
            pred.setX2(self.__limitBetween(pred.getX2() + int((random.random() - 0.5) * 2 * value), 0, imageShape[1] - 1))
            pred.setY1(self.__limitBetween(pred.getY1() + int((random.random() - 0.5) * 2 * value), 0, imageShape[0] - 1))
            pred.setY2(self.__limitBetween(pred.getY2() + int((random.random() - 0.5) * 2 * value), 0, imageShape[0] - 1))
            if pred.getX1() >= pred.getX2() or pred.getY1() >= pred.getY2():
                continue
            pred.setProb(random.random() * (self.fakeProbRange[1] - self.fakeProbRange[0]) + self.fakeProbRange[0])
            predictedBBoxes.append(pred)
        return predictedBBoxes

    def predictForFrame(self, frameNr):
        return self.getPredictedBoxesForFrame(frameNr) + self.getFakePredictionsForFrame(frameNr)

