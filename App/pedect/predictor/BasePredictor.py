from abc import abstractmethod

class BasePredictor:

    @abstractmethod
    def predictForFrame(self, frameNr):
        raise NotImplementedError

