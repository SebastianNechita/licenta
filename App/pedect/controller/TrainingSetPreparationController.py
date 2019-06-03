from PySide2.QtGui import QStandardItemModel
from PySide2.QtWidgets import QListView, QPushButton

from pedect.design.uiHelper import deselectAllFromModel, selectVideosFromModel, populateModel, getCheckedVideos
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
        videoList = getCheckedVideos(self.videosListModel)
        self.service.prepareTrainingSet(videoList)