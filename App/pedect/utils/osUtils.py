import os
import shutil

def emptyDirectory(path):
    if os.path.exists(path): # if it exist already
        shutil.rmtree(path)
    os.makedirs(path)

def createDirectoryIfNotExists(savePath):
    dir = "."
    for el in savePath:
        dir = os.path.join(dir, el)
        if not os.path.exists(dir):
            os.makedirs(dir)

def getPathFromList(savePath):
    dir = "."
    for el in savePath:
        dir = os.path.join(dir, el)
    return dir