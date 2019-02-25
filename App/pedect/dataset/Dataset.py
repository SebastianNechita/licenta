import os


class Dataset:
    def __init__(self, baseDir):
        self.baseDir = baseDir

    def getVideoPath(self, videoSet, videoNr):
        return os.path.join(self.baseDir, videoSet, videoNr + ".seq")

    def getAnnotationsPath(self, videoSet, videoNr):
        return os.path.join(self.baseDir, "annotations", videoSet, videoNr + ".vbb")
