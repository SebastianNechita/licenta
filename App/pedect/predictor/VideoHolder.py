from pedect.dataset import Dataset
from pedect.dataset.CaltechDataset import CaltechDataset
from pedect.input.inputProcessing import read_seq
from pedect.utils.constants import CALTECH_DIR


class VideoHolder:

    def __init__(self, chosenDataset, setName: str, videoNr: str):
        if isinstance(chosenDataset, str):
            chosenDataset = findDatasetByName(chosenDataset)
        self.chosenDataset = chosenDataset
        self.setName = setName
        self.videoNr = videoNr
        video_path = chosenDataset.getVideoPath(setName, videoNr)
        self.video = read_seq(video_path)

    def getLength(self):
        return len(self.video)

    def getVideo(self):
        return self.video

    def getFrame(self, frameNr):
        return self.video[frameNr]


def findDatasetByName(datasetName: str) -> Dataset:
    if datasetName == 'caltech':
        dataset = CaltechDataset(CALTECH_DIR)
    else:
        raise Exception("Error no such dataset defined in this function!")
    return dataset