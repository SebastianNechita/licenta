import shutil

from pedect.dataset.datasetHelper import findDatasetByName
from pedect.utils.constants import *
from pedect.input.inputProcessing import *
from PIL import Image
from tqdm import tqdm

class ConverterToImagesYOLOv3:

    def __init__(self, textPattern: str):
        self.textPattern = textPattern
        self.csvSet = set()
        self.possibleLabels = {}
        self.textPattern = ""
        self.uId = 0

    def clearDirectory(self):
        for the_file in os.listdir(FINAL_IMAGES_DIR):
            file_path = os.path.join(FINAL_IMAGES_DIR, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        self.csvSet = set()

    def saveImagesFromGroundTruth(self, datasetName: str, setName: str, videoNr: str):
        print("Saving images from %s set %s video %s" % (datasetName, setName, videoNr))
        chosenDataset = findDatasetByName(datasetName)

        video_path = chosenDataset.getVideoPath(setName, videoNr)
        annotations_path = chosenDataset.getAnnotationsPath(setName, videoNr)
        v = read_seq(video_path)
        annotations = read_vbb(annotations_path)
        for i in tqdm(range(len(v))):
            img = Image.fromarray(v[i])
            anns = getAnnotationsForFrame(i, annotations)
            imageName = self.textPattern % (datasetName, setName, videoNr, i)
            imgPath = os.path.join(FINAL_IMAGES_DIR, imageName)
            img.save(imgPath)
            string = imgPath
            for ann in anns:
                x1 = int(ann["pos"][0])
                y1 = int(ann["pos"][1])
                x2 = int(ann["pos"][2]) + x1
                y2 = int(ann["pos"][3]) + y1
                if x1 >= x2 or y1 >= y2:
                    continue
                label = ann["lbl"]
                if label not in self.possibleLabels:
                    self.possibleLabels[label] = self.uId
                    self.uId = self.uId + 1
                string += " %d,%d,%d,%d,%d" % (x1, y1, x2, y2, self.possibleLabels[label])

            self.csvSet.add(string)


    def writeAnnotationsFile(self):
        try:
            os.remove(ANNOTATIONS_FILE)
        except OSError:
            pass
        f = open(ANNOTATIONS_FILE, "a+")
        l = []
        for string in self.csvSet:
            l.append(string)
        l.sort()
        for string in l:
            f.write(string + "\n")
        f.close()

        try:
            os.remove(LABELS_FILE)
        except OSError:
            pass
        f = open(LABELS_FILE, "w+")
        inverseLabels = {v: k for k, v in self.possibleLabels.items()}
        for label in inverseLabels.keys():
            f.write("%s\n" % inverseLabels[label])
        f.close()
        self.csvSet = set()
