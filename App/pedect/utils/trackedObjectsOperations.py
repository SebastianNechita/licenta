import math

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

def positionBetween(b1: tuple, b2: tuple, percent: float) -> tuple:
    a = []
    for i in range(4):
        a.append(b1[i] * percent + b2[i] * (1 - percent))
    return tuple(a)

def moveOrDestroyTrackedObjects(activeObjects, predictedBBoxes, surviveMovePercent, surviveThreshold):
    objectsProbabilities = {}
    if len(predictedBBoxes) == 0:
        activeObjects = {}
    else:
        survivingObjects = {}
        for activeId, v in activeObjects.items():
            intersections = [math.sqrt(IOU(v.getPos(), predBox.getPos()) * predBox.getProb()) if v.getLabel() == predBox.getLabel() else 0
                             for predBox in predictedBBoxes]
            maxValue = max(intersections)
            bestPos = intersections.index(maxValue)
            predBox = predictedBBoxes[bestPos].getPos()
            if intersections[bestPos] >= surviveThreshold:
                survivingObjects[activeId] = v
                objectsProbabilities[activeId] = maxValue
                v.setPos(positionBetween(predBox, v.getPos(), surviveMovePercent))
        activeObjects = survivingObjects
    return activeObjects, objectsProbabilities


def createAndDestroyTrackedObjects(tracker, image, activeObjects, predictedBBoxes, createThreshold, removeThreshold,
                                   frameNr, probabilitiesDictionary):
    newObjects = {}
    for predBox in predictedBBoxes:
        box = predBox.getPos()
        prob = predBox.getProb()
        # print(predBox.getPos(), prob, box)
        # print(createThreshold, removeThreshold, frameNr)
        if prob >= createThreshold:
            newId = IdGenerator.getStringId()
            activeObjects = {k: v for k, v in activeObjects.items() if IOU(v.getPos(), box) < removeThreshold}
            newObjects[newId] = TrackedObject(box, frameNr, predBox.getLabel())
            probabilitiesDictionary[newId] = prob
            tracker.track(newId, image, box)
    for k, v in newObjects.items():
        activeObjects[k] = v
    return activeObjects

def removeOldObjects(activeObjects, frameNr, maxAge):
    return {k: v for k, v in activeObjects.items() if frameNr - v.getFrameCreated() < maxAge}