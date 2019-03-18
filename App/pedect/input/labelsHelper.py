from pedect.utils.constants import LABELS_FILE


def getAllPossibleLabelsDictionary(labelsFile = LABELS_FILE) -> dict:
    f = open(labelsFile, "r")
    labels = {}
    i = 0
    for line in f.readlines():
        labels[line.split('\n')[0]] = i
        i = i + 1
    f.close()
    return labels