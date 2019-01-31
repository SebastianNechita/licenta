from pedect.dataset.Dataset import Dataset

class CaltechDataset(Dataset):
    def __init__(self, baseDir="..\\Data\\caltech"):
        super().__init__(baseDir)

    def getVideoPath(self, videoSet, videoNr):
        return "%s\\%s\\%s.seq" % (self.baseDir, videoSet, videoNr)

    def getAnnotationsPath(self, videoSet, videoNr):
        return "%s\\annotations\\%s\\%s.vbb" % (self.baseDir, videoSet, videoNr)
