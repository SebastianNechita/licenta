import os


class Dataset:
    datasetName = "Dataset"
    def __init__(self, baseDir: str):
        self.baseDir = baseDir

    def getVideoPath(self, videoSet: str, videoNr: str):
        return os.path.join(self.baseDir, videoSet, videoNr + ".seq")

    def getAnnotationsPath(self, videoSet: str, videoNr: str):
        return os.path.join(self.baseDir, "annotations", videoSet, videoNr + ".vbb")


