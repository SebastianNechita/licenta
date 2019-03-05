import shutil

from pedect.utils.constants import *
from pedect.dataset.CaltechDataset import *
from pedect.dataset.DaimlerDataset import DaimlerDataset
from pedect.dataset.InriaDataset import InriaDataset
from pedect.input.inputProcessing import *
from PIL import Image
from tqdm import tqdm

class ConverterToImagesRetinaNet:
    csvSet = set()
    possibleLabels = set()
    textPattern = ""

    def __init__(self, textPattern):
        self.textPattern = textPattern

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

    def saveImages(self, datasetName, setName, videoNr):
        print("Saving images from %s set %s video %s" % (datasetName, setName, videoNr))
        if datasetName == 'caltech':
            chosenDataset = CaltechDataset(CALTECH_DIR)
        elif datasetName == 'daimler':
            chosenDataset = DaimlerDataset(DAIMLER_DIR)
        elif datasetName == 'inria':
            chosenDataset = InriaDataset(INRIA_DIR)
        else:
            raise Exception("No such dataset!")

        video_path = chosenDataset.getVideoPath(setName, videoNr)
        annotations_path = chosenDataset.getAnnotationsPath(setName, videoNr)
        v = read_seq(video_path)
        annotations = read_vbb(annotations_path)
        for i in tqdm(range(len(v))):
            print(v[i])
            img = Image.fromarray(v[i])
            anns = getAnnotationsForFrame(i, annotations)
            imageName = self.textPattern % (datasetName, setName, videoNr, i)
            imgPath = os.path.join(FINAL_IMAGES_DIR, imageName)
            img.save(imgPath)
            for ann in anns:
                x1 = int(ann["pos"][0])
                y1 = int(ann["pos"][1])
                x2 = int(ann["pos"][2]) + x1
                y2 = int(ann["pos"][3]) + y1
                if x1 >= x2 or y1 >= y2:
                    continue
                label = ann["lbl"]
                self.csvSet.add("%s,%d,%d,%d,%d,%s" % (imageName, x1, y1, x2, y2, label))
                self.possibleLabels.add(label)
            if len(anns) == 0:
                self.csvSet.add("%s,,,,," % imageName)

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
        i = 0
        for label in self.possibleLabels:
            f.write("%s, %d\n" % (label, i))
            i = i + 1
        f.close()
        self.csvSet = set()

class ConverterToImagesYoloV3:
    csvSet = set()
    possibleLabels = {}
    textPattern = ""
    uId = 0

    def __init__(self, textPattern):
        self.textPattern = textPattern

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

    def saveImages(self, datasetName, setName, videoNr):
        print("Saving images from %s set %s video %s" % (datasetName, setName, videoNr))
        if datasetName == 'caltech':
            chosenDataset = CaltechDataset(CALTECH_DIR)
        elif datasetName == 'daimler':
            chosenDataset = DaimlerDataset(DAIMLER_DIR)
        elif datasetName == 'inria':
            chosenDataset = InriaDataset(INRIA_DIR)
        else:
            raise Exception("No such dataset!")

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

YOLO = ConverterToImagesYoloV3
RETINANET = ConverterToImagesRetinaNet

def convertVideosToImages(videoList, datasetOption):
    textPattern = "%s-%s-%s-%s.jpg"
    converter = datasetOption(textPattern)
    converter.clearDirectory()
    for video in videoList:
        converter.saveImages(video[0], video[1], video[2])
    converter.writeAnnotationsFile()