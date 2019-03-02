import os
import random

from pedect.utils.osUtils import emptyDirectory


class Evaluator:
    def __init__(self, config, predictor, groundTruthPredictor, maxFrames):
        self.predictor = predictor
        self.predictor.config = config
        self.groundTruthPredictor = groundTruthPredictor
        self.maxFrames = maxFrames
        self.computed = False
        self.result = 0

    def evaluate(self):
        if self.computed:
            return self.result
        s = random.getstate()
        random.seed(3)
        basePath = os.path.join(os.getcwd())
        gtPath = os.path.join(basePath, 'ground-truth')
        predictedPath = os.path.join(basePath, 'predicted')
        emptyDirectory(predictedPath)
        emptyDirectory(gtPath)
        for frameNr in range(min(self.groundTruthPredictor.getLength(), self.maxFrames)):
            groundTruthObjects = self.groundTruthPredictor.predictForFrame(frameNr)
            predictedObjects = self.predictor.predictForFrame(frameNr)
            f = open(os.path.join(gtPath, str(frameNr) + ".txt"), "a+")
            [f.write("%s %d %d %d %d\n" % (o.getLabel(), o.getX1(), o.getY1(), o.getX2(), o.getY2()))
             for o in groundTruthObjects]
            f.close()
            f = open(os.path.join(predictedPath, str(frameNr) + ".txt"), "a+")
            [f.write("%s %f %d %d %d %d\n" % (o.getLabel(), o.getProb(), o.getX1(), o.getY1(), o.getX2(), o.getY2()))
             for o in predictedObjects]
            f.close()
        command = 'python mAP\\main.py -q -np'
        result = os.popen(command).read()
        self.result = float(result[5:-2])
        random.setstate(s)
        self.computed = True
        print(self.result)

        return self.result
