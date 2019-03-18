import os
import shutil


def emptyDirectory(path: str):
    if os.path.exists(path):  # if it exist already
        shutil.rmtree(path)
    os.makedirs(path)


def createDirectoryIfNotExists(savePath: str):
    os.makedirs(savePath, exist_ok=True)

# def getPathFromList(savePath):
#     dir = "."
#     for el in savePath:
#         dir = os.path.join(dir, el)
#     return dir