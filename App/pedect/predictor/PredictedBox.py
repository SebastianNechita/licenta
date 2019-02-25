class PredictedBox:
    __x1, __y1, __x2, __y2 = 0, 0, 0, 0
    __label = ""
    __prob = 0.0

    def __init__(self, x1, y1, x2, y2, label, prob):
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

    def setX1(self, x1):
        self.__x1 = x1

    def setY1(self, y1):
        self.__y1 = y1

    def setX2(self, x2):
        self.__x2 = x2

    def setY2(self, y2):
        self.__y2 = y2

    def setLabel(self, label):
        self.__label = label

    def setProb(self, prob):
        self.__prob = prob

