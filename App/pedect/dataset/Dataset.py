class Dataset:
    def __init__(self, baseDir):
        self.baseDir = baseDir

    def getVideoPath(self, videoSet, videoNr):
        raise Exception("Not implemented!")

    def getAnnotationsPath(self, videoSet, videoNr):
        raise Exception("Not implemented!")