from PySide2.QtCore import Qt
from PySide2.QtGui import QStandardItemModel, QStandardItem
from PySide2.QtWidgets import QListView, QPushButton, QLineEdit

from pedect.config.BasicConfig import getConfigFromTrainId
from pedect.controller.TrainIdsController import TrainIdsController
from pedect.design.uiHelper import deselectAllFromModel, selectVideosFromModel, populateModel
from pedect.service.Service import Service


class EvaluatingController:

    def __init__(self, service: Service, trainIdsController: TrainIdsController):
        self.service = service
        self.window = None
        self.trainIdListViewModel = None
        self.trainIdListView = None
        self.videosListModel = None
        self.resultEvaluationMAPLineEdit = None
        self.trainIdsController = trainIdsController


    def setUp(self, window):
        self.window = window

        listView = window.findChild(QListView, 'allVideosListViewEvaluation')
        self.videosListModel = QStandardItemModel(listView)
        listView.setModel(self.videosListModel)
        populateModel(self.videosListModel, self.service.getAllVideoTuples())

        chooseDefaultButton = window.findChild(QPushButton, 'chooseDefaultButtonEvaluation')
        deselectAllButton = window.findChild(QPushButton, 'deselectAllButtonEvaluation')
        evaluateButton = window.findChild(QPushButton, 'evaluateButton')
        self.resultEvaluationMAPLineEdit = window.findChild(QLineEdit, 'resultEvaluationMAPLineEdit')

        deselectAllButton.clicked.connect(lambda: deselectAllFromModel(self.videosListModel))
        chooseDefaultButton.clicked.connect(lambda: selectVideosFromModel(self.videosListModel,
                                                                          self.service.getEvaluationVideoList()))
        evaluateButton.clicked.connect(self.__evaluate)


    def __evaluate(self):
        trainId = self.trainIdsController.getSelectedTrainId()
        config = getConfigFromTrainId(trainId)
        trainList = []
        for i in range(self.videosListModel.rowCount()):
            item = self.videosListModel.item(i, 0)
            if isinstance(item, QStandardItem):
                if item.checkState() == Qt.CheckState.Checked:
                    trainList.append(tuple(item.data(1)))
        result = self.service.evaluatePredictor(config, trainList)
        self.resultEvaluationMAPLineEdit.setText(str(result['mAP']))




