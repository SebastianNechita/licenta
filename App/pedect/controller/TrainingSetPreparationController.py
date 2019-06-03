from PySide2.QtCore import Qt
from PySide2.QtGui import QStandardItemModel, QStandardItem
from PySide2.QtWidgets import QListView, QPushButton

from pedect.design.uiHelper import deselectAllFromModel, selectVideosFromModel, populateModel
from pedect.service.Service import Service


class TrainingSetPreparationController:

    def __init__(self, service: Service):
        self.service = service
        self.videosListModel = None
        self.window = None

    def setUp(self, window):
        self.window = window
        listView = window.findChild(QListView, 'allVideosListView')
        self.videosListModel = QStandardItemModel(listView)
        listView.setModel(self.videosListModel)
        populateModel(self.videosListModel, self.service.getAllVideoTuples())

        chooseDefaultButton = window.findChild(QPushButton, 'chooseDefaultButton')
        deselectAllButton = window.findChild(QPushButton, 'deselectAllButton')
        prepareTrainingSetButton = window.findChild(QPushButton, 'prepareTrainingSetButton')

        deselectAllButton.clicked.connect(lambda: deselectAllFromModel(self.videosListModel))
        chooseDefaultButton.clicked.connect(lambda: selectVideosFromModel(self.videosListModel,
                                                                          self.service.getTrainingVideoList()))
        prepareTrainingSetButton.clicked.connect(self.__prepareTrainingSet)


    def __prepareTrainingSet(self):
        trainList = []
        for i in range(self.videosListModel.rowCount()):
            item = self.videosListModel.item(i, 0)
            if isinstance(item, QStandardItem):
                if item.checkState() == Qt.CheckState.Checked:
                    trainList.append(tuple(item.data(1)))
        self.service.prepareTrainingSet(trainList)