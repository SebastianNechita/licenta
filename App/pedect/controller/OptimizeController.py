from PySide2.QtCore import Qt
from PySide2.QtGui import QStandardItemModel, QStandardItem
from PySide2.QtWidgets import QListView, QPushButton, QLineEdit, QCheckBox, QTableView

from pedect.config.BasicConfig import getConfigFromTrainId
from pedect.controller.TrainIdsController import TrainIdsController
from pedect.design.uiHelper import showSuccess, showError, deselectAllFromModel, selectVideosFromModel, populateModel, \
    between0And1, getCheckedVideos
from pedect.service.Service import Service


class OptimizeController:
    def __init__(self, service: Service, trainIdsController: TrainIdsController):
        self.service = service
        self.trainIdsController = trainIdsController
        self.window = None
        self.videosListModel = None
        self.ctFromTB = None
        self.rtFromTB = None
        self.stFromTB = None
        self.smpFromTB = None
        self.mspFromTB = None
        self.ctToTB = None
        self.rtToTB = None
        self.stToTB = None
        self.smpToTB = None
        self.mspToTB = None
        self.trackerTypesOptimizeListView = None
        self.cachedCheckBox = None
        self.maxNoFramesTextBox = None
        self.stepSizeTextBox = None
        self.findBestConfigurationsButton = None
        self.bestConfigurationsListView = None
        self.bestConfigurationsListViewModel = None
        self.trackerTypesOptimizeListViewModel = None

    def setUp(self, window):
        self.window = window
        listView = window.findChild(QListView, 'allVideosListViewOptimization')
        self.videosListModel = QStandardItemModel(listView)
        listView.setModel(self.videosListModel)
        populateModel(self.videosListModel, self.service.getAllVideoTuples())


        chooseDefaultButton = window.findChild(QPushButton, 'chooseDefaultButtonOptimization')
        deselectAllButton = window.findChild(QPushButton, 'deselectAllButtonOptimization')

        self.ctFromTB = window.findChild(QLineEdit, 'ctFromTB')
        self.rtFromTB = window.findChild(QLineEdit, 'rtFromTB')
        self.stFromTB = window.findChild(QLineEdit, 'stFromTB')
        self.smpFromTB = window.findChild(QLineEdit, 'smpFromTB')
        self.mspFromTB = window.findChild(QLineEdit, 'mspFromTB')

        self.ctToTB = window.findChild(QLineEdit, 'ctToTB')
        self.rtToTB = window.findChild(QLineEdit, 'rtToTB')
        self.stToTB = window.findChild(QLineEdit, 'stToTB')
        self.smpToTB = window.findChild(QLineEdit, 'smpToTB')
        self.mspToTB = window.findChild(QLineEdit, 'mspToTB')

        self.trackerTypesOptimizeListView = window.findChild(QListView, 'trackerTypesOptimizeListView')
        self.cachedCheckBox = window.findChild(QCheckBox, 'cachedCheckBox')
        self.maxNoFramesTextBox = window.findChild(QLineEdit, 'maxNoFramesTextBox')
        self.stepSizeTextBox = window.findChild(QLineEdit, 'stepSizeTextBox')

        self.findBestConfigurationsButton = window.findChild(QPushButton, 'findBestConfigurationsButton')

        self.bestConfigurationsListView = window.findChild(QTableView, 'bestConfigurationsTableView')
        self.bestConfigurationsListViewModel = QStandardItemModel(self.bestConfigurationsListView)
        self.bestConfigurationsListView.setModel(self.bestConfigurationsListViewModel)
        self.trackerTypesOptimizeListViewModel = QStandardItemModel(self.trackerTypesOptimizeListView)
        self.trackerTypesOptimizeListView.setModel(self.trackerTypesOptimizeListViewModel)
        for trackerType in self.service.getAllAvailableTrackerTypes():
            print(trackerType)
            item = QStandardItem(trackerType)
            item.setCheckable(True)
            self.trackerTypesOptimizeListViewModel.appendRow(item)

        deselectAllButton.clicked.connect(lambda: deselectAllFromModel(self.videosListModel))
        chooseDefaultButton.clicked.connect(lambda: selectVideosFromModel(self.videosListModel,
                                                                          self.service.getTuningVideoList()))
        self.findBestConfigurationsButton.clicked.connect(self.__findBestConfigurations)


    def __findBestConfigurations(self):
        try:
            ctFrom = between0And1(float(self.ctFromTB.text()))
            ctTo = between0And1(float(self.ctToTB.text()))
            rtFrom = between0And1(float(self.rtFromTB.text()))
            rtTo = between0And1(float(self.rtToTB.text()))
            stFrom = between0And1(float(self.stFromTB.text()))
            stTo = between0And1(float(self.stToTB.text()))
            smpFrom = between0And1(float(self.smpFromTB.text()))
            smpTo = between0And1(float(self.smpToTB.text()))
            mspFrom = between0And1(float(self.mspFromTB.text()))
            mspTo = between0And1(float(self.mspToTB.text()))

            ctRange = (ctFrom, ctTo)
            rtRange = (rtFrom, rtTo)
            stRange = (stFrom, stTo)
            smpRange = (smpFrom, smpTo)
            mspRange = (mspFrom, mspTo)

            videoList = getCheckedVideos(self.videosListModel)

            trackerTypeList = []
            for i in range(self.trackerTypesOptimizeListViewModel.rowCount()):
                item = self.trackerTypesOptimizeListViewModel.item(i, 0)
                if isinstance(item, QStandardItem):
                    if item.checkState() == Qt.CheckState.Checked:
                        trackerTypeList.append(item.text())

            if self.cachedCheckBox.checkState() == Qt.CheckState.Checked:
                trackerTypeList = ["cached %s" % x for x in trackerTypeList]

            maxNoFrames = int(self.maxNoFramesTextBox.text())
            stepSize = between0And1(float(self.stepSizeTextBox.text()))
            trainId = self.trainIdsController.getSelectedTrainId()
            config = getConfigFromTrainId(trainId)
            self.service.config = config
            titles, rows = self.service.optimizeTrackerConfig("temp.txt", trackerTypeList, ctRange, rtRange, stRange,
                                                              smpRange, mspRange, videoList, None, maxNoFrames, True,
                                                              stepSize)
            self.bestConfigurationsListViewModel.removeRows(0, self.bestConfigurationsListViewModel.rowCount())
            self.bestConfigurationsListViewModel.setHorizontalHeaderLabels(titles)
            for row in rows:
                self.bestConfigurationsListViewModel.appendRow([QStandardItem(str(el)) for el in row])
            showSuccess("YES!")
        except Exception as e:
            showError(str(e))




