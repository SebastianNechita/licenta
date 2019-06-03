import os

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QPushButton, QCheckBox, QLineEdit

from pedect.config.BasicConfig import getConfigFromTrainId, saveConfiguration
from pedect.controller.TrainIdsController import TrainIdsController
from pedect.design.uiHelper import showSuccess, showError
from pedect.service.Service import Service


class TrainingController:

    def __init__(self, service: Service, trainIdsController: TrainIdsController):
        self.service = service
        self.trainIdsController = trainIdsController

        self.window = None
        self.listViewModel = None
        self.modelNameTB = None
        self.inputShapeTB1 = None
        self.inputShapeTB2 = None
        self.freezeNoEpochsTB = None
        self.noFreezeNoEpochsTB = None
        self.validationSplitTB = None
        self.freezeBatchSizeTB = None
        self.noFreezeBatchSizeTB = None
        self.preTrainedModelPathTB = None
        self.checkpointPeriodTB = None
        self.initialLRTB = None
        self.alreadyTrainedEpochsTB = None
        self.loadPretrainedCheckbox = None
        self.isTinyCheckbox = None
        self.saveConfigurationButton = None
        self.saveConfigurationAndTrainButton = None
        self.retrainCheckBox = None


    def setUp(self, window):


        self.modelNameTB = window.findChild(QLineEdit, 'modelNameTB')
        self.inputShapeTB1 = window.findChild(QLineEdit, 'inputShapeTB1')
        self.inputShapeTB2 = window.findChild(QLineEdit, 'inputShapeTB2')
        self.freezeNoEpochsTB = window.findChild(QLineEdit, 'freezeNoEpochsTB')
        self.noFreezeNoEpochsTB = window.findChild(QLineEdit, 'noFreezeNoEpochsTB')
        self.validationSplitTB = window.findChild(QLineEdit, 'validationSplitTB')
        self.freezeBatchSizeTB = window.findChild(QLineEdit, 'freezeBatchSizeTB')
        self.noFreezeBatchSizeTB = window.findChild(QLineEdit, 'noFreezeBatchSizeTB')
        self.preTrainedModelPathTB = window.findChild(QLineEdit, 'preTrainedModelPathTB')
        self.checkpointPeriodTB = window.findChild(QLineEdit, 'checkpointPeriodTB')
        self.initialLRTB = window.findChild(QLineEdit, 'initialLRTB')
        self.alreadyTrainedEpochsTB = window.findChild(QLineEdit, 'alreadyTrainedEpochsTB')
        self.loadPretrainedCheckbox = window.findChild(QCheckBox, 'loadPretrainedCheckbox')
        self.isTinyCheckbox = window.findChild(QCheckBox, 'isTinyCheckbox')
        self.saveConfigurationButton = window.findChild(QPushButton, 'saveConfigurationButton')
        self.saveConfigurationAndTrainButton = window.findChild(QPushButton, 'saveConfigurationAndTrainButton')
        self.retrainCheckBox = window.findChild(QCheckBox, 'retrainCheckBox')

        self.saveConfigurationButton.clicked.connect(self.__uiToModel)
        self.saveConfigurationAndTrainButton.clicked.connect(self.__saveAndTrain)
        self.trainIdsController.listView.clicked.connect(self.__modelToUi)


    def __saveAndTrain(self):
        if not self.__uiToModel():
            return
        trainId = self.trainIdsController.getSelectedTrainId()
        config = getConfigFromTrainId(trainId)
        if self.retrainCheckBox.checkState() == Qt.CheckState.Checked:
            print("Retraining!")
            self.service.retrain(config)
        else:
            print("Training!")
            self.service.train(config)
        self.__modelToUi()
        showSuccess("Training complete!")


    def __modelToUi(self):
        trainId = self.trainIdsController.getSelectedTrainId()
        config = getConfigFromTrainId(trainId)
        print(config)
        self.modelNameTB.setText(config.modelName)
        self.inputShapeTB1.setText(str(config.inputShape[0]))
        self.inputShapeTB2.setText(str(config.inputShape[1]))
        self.freezeNoEpochsTB.setText(str(config.freezeNoEpochs))
        self.noFreezeNoEpochsTB.setText(str(config.noFreezeNoEpochs))
        self.validationSplitTB.setText(str(config.validationSplit))
        self.freezeBatchSizeTB.setText(str(config.freezeBatchSize))
        self.noFreezeBatchSizeTB.setText(str(config.noFreezeBatchSize))
        self.preTrainedModelPathTB.setText(config.preTrainedModelPath)
        self.checkpointPeriodTB.setText(str(config.checkpointPeriod))
        self.initialLRTB.setText(str(config.initialLR))
        self.alreadyTrainedEpochsTB.setText(str(config.alreadyTrainedEpochs))
        self.isTinyCheckbox.setCheckState(Qt.CheckState(2 if config.isTiny else 0))
        self.loadPretrainedCheckbox.setCheckState(Qt.CheckState(2 if config.loadPreTrained else 0))

        self.preTrainedModelPathTB.setDisabled(True if config.alreadyTrainedEpochs > 0 else False)
        self.inputShapeTB1.setDisabled(True if config.alreadyTrainedEpochs > 0 else False)
        self.inputShapeTB2.setDisabled(True if config.alreadyTrainedEpochs > 0 else False)
        self.modelNameTB.setDisabled(True if config.alreadyTrainedEpochs > 0 else False)
        self.isTinyCheckbox.setDisabled(True if config.alreadyTrainedEpochs > 0 else False)
        self.loadPretrainedCheckbox.setDisabled(True if config.alreadyTrainedEpochs > 0 else False)
        self.alreadyTrainedEpochsTB.setDisabled(True)

    def __uiToModel(self) -> bool:
        try:
            trainId = self.trainIdsController.getSelectedTrainId()
            config = getConfigFromTrainId(trainId)
            config.modelName = self.modelNameTB.text()
            config.inputShape = (int(self.inputShapeTB1.text()), int(self.inputShapeTB2.text()))
            assert config.inputShape[0] % 32 == 0, "The first dimension of the input shape must be divisible by 32!"
            assert config.inputShape[1] % 32 == 0, "The second dimension of the input shape must be divisible by 32!"

            config.freezeNoEpochs = int(self.freezeNoEpochsTB.text())
            config.noFreezeNoEpochs = int(self.noFreezeNoEpochsTB.text())
            config.validationSplit = float(self.validationSplitTB.text())
            config.freezeBatchSize = int(self.freezeBatchSizeTB.text())
            config.noFreezeBatchSize = int(self.noFreezeBatchSizeTB.text())
            config.preTrainedModelPath = self.preTrainedModelPathTB.text()
            config.checkpointPeriod = int(self.checkpointPeriodTB.text())
            config.initialLR = float(self.initialLRTB.text())
            config.alreadyTrainedEpochs = int(self.alreadyTrainedEpochsTB.text())
            config.isTiny = True if self.isTinyCheckbox.checkState() == Qt.CheckState.Checked else False
            config.loadPreTrained = True if self.loadPretrainedCheckbox.checkState() == Qt.CheckState.Checked else False
            saveConfiguration(config)
            showSuccess("Saved!")
            self.__modelToUi()
            return True
        except Exception as e:
            showError(str(e))
            return False





