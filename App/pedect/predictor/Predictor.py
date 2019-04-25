from abc import abstractmethod

class Predictor:

    @abstractmethod
    def predictForFrame(self, frameNr: int):
        raise NotImplementedError

    @abstractmethod
    def startNewPrediction(self):
        raise NotImplementedError

