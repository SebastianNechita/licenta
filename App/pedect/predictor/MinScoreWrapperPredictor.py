from pedect.predictor.Predictor import Predictor


class MinScoreWrapperPredictor(Predictor):

    def __init__(self, predictor: Predictor, minScore: float):
        self.predictor = predictor
        self.minScore = minScore

    def predictForFrame(self, frameNr: int):
        prediction = self.predictor.predictForFrame(frameNr)
        return [pred for pred in prediction if pred.getProb() >= self.minScore]

    # def finishPrediction(self):
    #     self.predictor.finishPrediction()
