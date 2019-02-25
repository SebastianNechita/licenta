class BasicConfig:
    possibleLabels = {'people': (255, 0, 0), 'person-fa': (0, 0, 255), 'person': (0, 255, 0)}
    batchSize = 1
    steps = 100
    nrEpochs = 100
    imageMaxSide = 600
    backbone = "mobilenet128_1.0"
    createThreshold = 0.9
    removeThreshold = 0.5
    surviveThreshold = 0.2
    surviveMovePercent = 0.0
    maxAge = 100

    def configName(self):
        return "BasicConfig"

    def __str__(self):
        return """
        %s
        Possible labels = %s
        Batch size = %d
        Steps per epoch = %d
        Number of epochs = %d
        Image max side = %d
        Backbone = %s
        Create Threshold = %f
        Remove Threshold = %f
        Survive Threshold = %f
        Survive Move Percent = %f
        Max Age = %d
        """\
               % (self.configName(),
                  str(self.possibleLabels),
                  self.batchSize,
                  self.steps,
                  self.nrEpochs,
                  self.imageMaxSide,
                  self.backbone,
                  self.createThreshold,
                  self.removeThreshold,
                  self.surviveThreshold,
                  self.surviveMovePercent,
                  self.maxAge
                  )
#
#
# if __name__ == '__main__':
#     config = BasicConfig()
#     print(config)

