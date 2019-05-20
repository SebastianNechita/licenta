from pedect.dataset import Dataset
from pedect.dataset.CaltechDataset import CaltechDataset
from pedect.utils.constants import CALTECH_DIR


def findDatasetByName(datasetName: str) -> Dataset:
    if datasetName == 'caltech':
        dataset = CaltechDataset(CALTECH_DIR)
    else:
        raise Exception("Error no such dataset defined in this function!")
    return dataset