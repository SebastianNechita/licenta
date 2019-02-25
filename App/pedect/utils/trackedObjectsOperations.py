from pedect.utils.IdGenerator import IdGenerator
from pedect.utils.metrics import IOU

class TrackedObject:
    def __init__(self, pos, frameCreated, label):
        self.__pos, self.__frameCreated, self.__label = pos, frameCreated, label

    def getPos(self):
        return self.__pos

    def getLabel(self):
        return self.__label

    def getFrameCreated(self):
        return self.__frameCreated

    def setPos(self, pos):
        self.__pos = pos

    def setLabel(self, label):
        self.__label = label

    def setFrameCreated(self, frameCreated):
        self.__frameCreated = frameCreated


def refreshTrackedObjects(tracker, image, activeObjects):
    imageRGB = image[:, :, ::-1]
    for k, v in activeObjects.items():
        activeObjects[k].setPos(tracker.track(k, imageRGB))
    return activeObjects

def positionBetween(b1, b2, percent):
    a = []
    for i in range(4):
        a.append(b1[i] * percent + b2[i] * (1 - percent))
    return a[0], a[1], a[2], a[3]

def moveOrDestroyTrackedObjects(activeObjects, predictedBBoxes, surviveMovePercent, surviveThreshold):
    if len(predictedBBoxes) == 0:
        activeObjects = {}
    else:
        survivingObjects = {}
        for activeId, v in activeObjects.items():
            intersections = [IOU(v.getPos(), predBox.getPos()) * predBox.getProb() if v.getLabel() == predBox.getLabel() else 0
                             for predBox in predictedBBoxes]
            bestPos = intersections.index(max(intersections))
            predBox = predictedBBoxes[bestPos].getPos()
            if intersections[bestPos] >= surviveThreshold:
                survivingObjects[activeId] = v
                v.setPos(positionBetween(predBox, v.getPos(), surviveMovePercent))
        activeObjects = survivingObjects
    return activeObjects


def createAndDestroyTrackedObjects(tracker, image, activeObjects, predictedBBoxes, createThreshold, removeThreshold,
                                   frameNr):
    for predBox in predictedBBoxes:
        box = predBox.getPos()
        prob = predBox.getProb()
        if prob >= createThreshold:
            newId = IdGenerator.getStringId()
            activeObjects = {k: v for k, v in activeObjects.items() if IOU(v.getPos(), box) < removeThreshold}
            activeObjects[newId] = TrackedObject(box, frameNr, predBox.getLabel())
            tracker.track(newId, image, box)
    return activeObjects

def removeOldObjects(activeObjects, frameNr, maxAge):
    return {k: v for k, v in activeObjects.items() if frameNr - v.getFrameCreated() < maxAge}