from PySide2.QtCore import Qt
from PySide2.QtGui import QStandardItem, QStandardItemModel
from PySide2.QtWidgets import QListView, QPushButton

from pedect.controller.TrainIdsController import TrainIdsController
from pedect.design.uiHelper import deselectAllFromModel, selectVideosFromModel, populateModel
from pedect.service.Service import Service


class GenerationController:
    def __init__(self, service: Service, trainIdsController: TrainIdsController):
        self.service = service
        self.trainIdsController = trainIdsController
        self.window = None
        self.videosListModel = None

    def setUp(self, window):
        self.window = window

        listView = window.findChild(QListView, 'allVideosListViewGeneration')
        self.videosListModel = QStandardItemModel(listView)
        listView.setModel(self.videosListModel)
        populateModel(self.videosListModel, self.service.getAllVideoTuples())

        chooseDefaultButton = window.findChild(QPushButton, 'chooseDefaultButtonGeneration')
        deselectAllButton = window.findChild(QPushButton, 'deselectAllButtonGeneration')

        deselectAllButton.clicked.connect(lambda: deselectAllFromModel(self.videosListModel))
        chooseDefaultButton.clicked.connect(lambda: selectVideosFromModel(self.videosListModel,
                                                                          self.service.getGenerationVideoList()))
