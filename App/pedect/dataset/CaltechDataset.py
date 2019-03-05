from pedect.dataset.Dataset import Dataset

class CaltechDataset(Dataset):
    datasetName = "caltech"
    def __init__(self, baseDir):
        super().__init__(baseDir)


