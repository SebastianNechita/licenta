from pedect.predictor.Predictor import Predictor


class MinScoreWrapperPredictor(Predictor):
    def __init__(self, predictor, minScore):
        self.predictor = predictor
        self.minScore = minScore

    def predictForFrame(self, frameNr):
        prediction = self.predictor.predictForFrame(frameNr)
        return [pred for pred in prediction if pred.getProb() >= self.minScore]
