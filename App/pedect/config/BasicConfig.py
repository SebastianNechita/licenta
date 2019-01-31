class BasicConfig:
    possibleLabels = {'people': (255, 0, 0), 'person-fa': (0, 0, 255), 'person': (0, 255, 0)}
    batchSize = 2
    stepsPerEpoch = 100
    nrEpochs = 10

    def configName(self):
        return "BasicConfig"

    def __str__(self):
        s = self.configName()
        s += "\nPossible labels = " + str(self.possibleLabels)
        s += "\nBatchs size = " + str(self.batchSize)
        s += "\nSteps per epoch = " + str(self.stepsPerEpoch)
        s += "\nNumber of epochs = " + str(self.nrEpochs)
        s += "\n"
        return s
#
#
# if __name__ == '__main__':
#     config = BasicConfig()
#     print(config)

