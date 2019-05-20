
class PredictedBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, label: str, prob: float):
        assert isinstance(x1, int)
        assert isinstance(y1, int)
        assert isinstance(x2, int)
        assert isinstance(y2, int)
        assert isinstance(label, str)
        assert (isinstance(prob, float) or isinstance(prob, int))
        self.__x1, self.__y1, self.__x2, self.__y2 = x1, y1, x2, y2
        self.__label = label
        self.__prob = prob

    def getX1(self):
        return self.__x1

    def getX2(self):
        return self.__x2

    def getY1(self):
        return self.__y1

    def getY2(self):
        return self.__y2

    def getPos(self):
        return self.__x1, self.__y1, self.__x2, self.__y2

    def getLabel(self):
        return self.__label

    def getProb(self):
        return self.__prob

    def setX1(self, x1: int):
        self.__x1 = x1

    def setY1(self, y1: int):
        self.__y1 = y1

    def setX2(self, x2: int):
        self.__x2 = x2

    def setY2(self, y2: int):
        self.__y2 = y2

    def setLabel(self, label: str):
        self.__label = label

    def setProb(self, prob: float):
        self.__prob = prob

    def __str__(self):
        return str((self.getX1(), self.getY1(), self.getX2(), self.getY2(), self.getProb(), self.getLabel()))
