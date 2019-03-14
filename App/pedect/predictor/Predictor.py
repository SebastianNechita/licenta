from abc import abstractmethod

class Predictor:

    @abstractmethod
    def predictForFrame(self, frameNr):
        raise NotImplementedError

