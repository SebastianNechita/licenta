import random

from tqdm import tqdm

from pedect.config.BasicConfig import BasicConfig
from pedect.utils.evaluator.Evaluator import Evaluator


class HyperParametersTuner:

    @staticmethod
    def findBestConfig(predictor, groundTruthPredictor, maxFrames, noIterations, ctRange, rtRange, stRange, smpRange):
        bestResult = (0, -1)
        for _ in tqdm(range(noIterations)):
            ct = random.random() * (ctRange[1] - ctRange[0]) + ctRange[0]
            rt = random.random() * (rtRange[1] - rtRange[0]) + rtRange[0]
            st = random.random() * (stRange[1] - stRange[0]) + stRange[0]
            smp = random.random() * (smpRange[1] - smpRange[0]) + smpRange[0]
            config = BasicConfig()
            config.createThreshold = ct
            config.removeThreshold = rt
            config.surviveThreshold = st
            config.surviveMovePercent = smp
            evaluator = Evaluator(config, predictor, groundTruthPredictor, maxFrames)
            result = evaluator.evaluate()
            if result >= bestResult[1]:
                bestResult = (config, result)

        return bestResult