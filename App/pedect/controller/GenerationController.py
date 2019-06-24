import traceback
from threading import Thread

from PySide2.QtCore import Qt
from PySide2.QtGui import QStandardItemModel
from PySide2.QtWidgets import QListView, QPushButton, QLineEdit, QCheckBox

from pedect.config.BasicConfig import getConfigFromTrainId, saveConfiguration
from pedect.controller.TrainIdsController import TrainIdsController
from pedect.design.uiHelper import deselectAllFromModel, selectVideosFromModel, populateModel, between0And1, \
    getCheckedVideos, ButtonEnablerManager, messageManager
from pedect.service.Service import Service


class GenerationController:
    def __init__(self, service: Service, trainIdsController: TrainIdsController):
        self.service = service
        self.trainIdsController = trainIdsController
        self.window = None
        self.videosListModel = None
        self.ctTB = None
        self.rtTH = None
        self.stTB = None
        self.smpTB = None
        self.mspTB = None
        self.imageGenerationSavePeriodTB = None
        self.maxNoObjectsPerFrameTB = None
        self.maxAgeTB = None
        self.trackerTypeTB = None
        self.cachedCheckBoxGeneration = None

    def setUp(self, window):
        self.window = window

        listView = window.findChild(QListView, 'allVideosListViewGeneration')
        self.videosListModel = QStandardItemModel(listView)
        listView.setModel(self.videosListModel)
        populateModel(self.videosListModel, self.service.getAllVideoTuples())

        chooseDefaultButton = window.findChild(QPushButton, 'chooseDefaultButtonGeneration')
        deselectAllButton = window.findChild(QPushButton, 'deselectAllButtonGeneration')

        self.ctTB = window.findChild(QLineEdit, 'ctTB')
        self.rtTH = window.findChild(QLineEdit, 'rtTH')
        self.stTB = window.findChild(QLineEdit, 'stTB')
        self.smpTB = window.findChild(QLineEdit, 'smpTB')
        self.mspTB = window.findChild(QLineEdit, 'mspTB')
        self.imageGenerationSavePeriodTB = window.findChild(QLineEdit, 'imageGenerationSavePeriodTB')
        self.maxNoObjectsPerFrameTB = window.findChild(QLineEdit, 'maxNoObjectsPerFrameTB')
        self.maxAgeTB = window.findChild(QLineEdit, 'maxAgeTB')
        self.trackerTypeTB = window.findChild(QLineEdit, 'trackerTypeTB')
        self.cachedCheckBoxGeneration = window.findChild(QCheckBox, 'cachedCheckBoxGeneration')

        saveConfigurationGenerateButton = window.findChild(QPushButton, 'saveConfigurationGenerateButton')
        playVideosButton = window.findChild(QPushButton, 'playVideosButton')
        generateNewTrainingDataButton = window.findChild(QPushButton, 'generateNewTrainingDataButton')



        saveConfigurationGenerateButton.clicked.connect(self.__uiToModel)
        playVideosButton.clicked.connect(self.__playVideos)
        generateNewTrainingDataButton.clicked.connect(self.__saveAndGenerateNewTrainingData)
        deselectAllButton.clicked.connect(lambda: deselectAllFromModel(self.videosListModel))
        chooseDefaultButton.clicked.connect(lambda: selectVideosFromModel(self.videosListModel,
                                                                          self.service.getGenerationVideoList()))

        self.trainIdsController.listView.clicked.connect(self.__modelToUi)

        ButtonEnablerManager.addButton(saveConfigurationGenerateButton)
        ButtonEnablerManager.addButton(playVideosButton)
        ButtonEnablerManager.addButton(generateNewTrainingDataButton)
        ButtonEnablerManager.addButton(deselectAllButton)
        ButtonEnablerManager.addButton(chooseDefaultButton)

    def __saveAndGenerateNewTrainingData(self):
        ButtonEnablerManager.setAllButtonsDisabledState(True)
        Thread(target=lambda: (self.__saveAndGenerateNewTrainingDataHelper(),
                               ButtonEnablerManager.setAllButtonsDisabledState(False))).start()

    def __saveAndGenerateNewTrainingDataHelper(self):
        try:
            if not self.__uiToModel():
                return
            videoList = getCheckedVideos(self.videosListModel)
            trainId = self.trainIdsController.getSelectedTrainId()
            config = getConfigFromTrainId(trainId)
            self.service.generateNewData(config, videoList)
            messageManager.success.emit("New data generated!")
        except Exception as e:
            messageManager.failure.emit(str(e))
            traceback.print_exc()


    def __playVideos(self):
        ButtonEnablerManager.setAllButtonsDisabledState(True)
        Thread(target=lambda: (self.__playVideosHelper(),
                               ButtonEnablerManager.setAllButtonsDisabledState(False))).start()

    def __playVideosHelper(self):
        videoList = getCheckedVideos(self.videosListModel)
        try:
            config = self.__getConfigFromUi()
            for videoTuple in videoList:
                self.service.playVideo(videoTuple, config)
        except Exception as e:
            messageManager.failure.emit(str(e))
            traceback.print_exc()

    def __modelToUi(self):
        trainId = self.trainIdsController.getSelectedTrainId()
        config = getConfigFromTrainId(trainId)
        self.ctTB.setText(str(config.createThreshold))
        self.rtTH.setText(str(config.removeThreshold))
        self.stTB.setText(str(config.surviveThreshold))
        self.smpTB.setText(str(config.surviveMovePercent))
        self.mspTB.setText(str(config.minScorePrediction))
        self.imageGenerationSavePeriodTB.setText(str(config.imageGenerationSavePeriod))
        self.maxNoObjectsPerFrameTB.setText(str(config.maxNrOfObjectsPerFrame))
        self.maxAgeTB.setText(str(config.maxAge))

        trackerTypeWords = config.trackerType.split(" ")
        self.cachedCheckBoxGeneration.setCheckState(Qt.CheckState(2 if len(trackerTypeWords) > 1 else 0))
        self.trackerTypeTB.setText(trackerTypeWords[len(trackerTypeWords) - 1])

    def __getConfigFromUi(self):
        trainId = self.trainIdsController.getSelectedTrainId()
        config = getConfigFromTrainId(trainId)
        config.createThreshold = between0And1(float(self.ctTB.text()))
        config.removeThreshold = between0And1(float(self.rtTH.text()))
        config.surviveThreshold = between0And1(float(self.stTB.text()))
        config.surviveMovePercent = between0And1(float(self.smpTB.text()))
        config.minScorePrediction = between0And1(float(self.mspTB.text()))
        config.imageGenerationSavePeriod = int(self.imageGenerationSavePeriodTB.text())
        config.maxNrOfObjectsPerFrame = int(self.maxNoObjectsPerFrameTB.text())
        config.maxAge = int(self.maxAgeTB.text())

        assert config.imageGenerationSavePeriod >= 1, "The image generation save period must be >= 1"
        assert config.maxNrOfObjectsPerFrame >= 1, "The max nr. of objects per frame must be >= 1"
        assert config.maxAge >= 1, "The max age must be >= 1"
        config.trackerType = self.trackerTypeTB.text()
        assert config.trackerType in self.service.getAllAvailableTrackerTypes(), "No such tracker type!"
        if self.cachedCheckBoxGeneration.checkState() == Qt.CheckState.Checked:
            config.trackerType = "cached " + config.trackerType
        return config

    def __uiToModel(self) -> bool:
        try:
            config = self.__getConfigFromUi()
            saveConfiguration(config)
            messageManager.success.emit("Saved!")
            self.__modelToUi()
            return True
        except Exception as e:
            messageManager.failure.emit(str(e))
            return False
