import os
import random

from pedect.utils.constants import MAX_VIDEO_LENGTH
from pedect.utils.osUtils import emptyDirectory
from tqdm import tqdm


class Evaluator:
    def __init__(self, predictors, groundTruthPredictors, maxFrames = MAX_VIDEO_LENGTH):
        self.predictors = predictors
        self.groundTruthPredictors = groundTruthPredictors
        self.maxFrames = maxFrames
        self.computed = False
        self.result = 0

    def evaluate(self, verbose = False):
        if self.computed:
            return self.result
        s = random.getstate()
        random.seed(3)
        basePath = os.path.join(os.getcwd())
        gtPath = os.path.join(basePath, 'ground-truth')
        predictedPath = os.path.join(basePath, 'predicted')
        emptyDirectory(predictedPath)
        emptyDirectory(gtPath)
        i = 0
        for predictor, groundTruthPredictor in zip(self.predictors, self.groundTruthPredictors):
            i = i + 1
            rangeToIterate = range(min(groundTruthPredictor.getLength(), self.maxFrames))
            if verbose:
                rangeToIterate = tqdm(rangeToIterate)
            for frameNr in rangeToIterate:
                fileName = "%d-%d.txt" % (i, frameNr)
                groundTruthObjects = groundTruthPredictor.predictForFrame(frameNr)
                predictedObjects = predictor.predictForFrame(frameNr)
                f = open(os.path.join(gtPath, fileName), "a+")
                [f.write("%s %d %d %d %d\n" % (o.getLabel(), o.getX1(), o.getY1(), o.getX2(), o.getY2()))
                 for o in groundTruthObjects]
                f.close()
                f = open(os.path.join(predictedPath, fileName), "a+")
                [f.write("%s %f %d %d %d %d\n" % (o.getLabel(), o.getProb(), o.getX1(), o.getY1(), o.getX2(), o.getY2()))
                 for o in predictedObjects]
                f.close()


        command = 'python mAP\\main.py -q -np'
        result = os.popen(command).read()
        self.result = float(result[5:-2])
        random.setstate(s)
        self.computed = True
        return self.result
