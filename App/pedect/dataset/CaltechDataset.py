from pedect.dataset.Dataset import Dataset
import os
class CaltechDataset(Dataset):
    def __init__(self, baseDir=os.path.join("..", "..", "Data", "caltech")):
        super().__init__(baseDir)

    def getVideoPath(self, videoSet, videoNr):
        return "%s\\%s\\%s.seq" % (self.baseDir, videoSet, videoNr)

    def getAnnotationsPath(self, videoSet, videoNr):
        return "%s\\annotations\\%s\\%s.vbb" % (self.baseDir, videoSet, videoNr)
