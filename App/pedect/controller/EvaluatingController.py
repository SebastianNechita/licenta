from threading import Thread

from PySide2.QtGui import QStandardItemModel
from PySide2.QtWidgets import QListView, QPushButton, QLineEdit

from pedect.config.BasicConfig import getConfigFromTrainId
from pedect.controller.TrainIdsController import TrainIdsController
from pedect.design.uiHelper import deselectAllFromModel, selectVideosFromModel, populateModel, getCheckedVideos, \
    ButtonEnablerManager
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
        ButtonEnablerManager.addButton(deselectAllButton)
        ButtonEnablerManager.addButton(chooseDefaultButton)
        ButtonEnablerManager.addButton(evaluateButton)



    def __evaluate(self):
        ButtonEnablerManager.setAllButtonsDisabledState(True)
        trainId = self.trainIdsController.getSelectedTrainId()
        config = getConfigFromTrainId(trainId)
        videoList = getCheckedVideos(self.videosListModel)
        Thread(target=lambda: (self.resultEvaluationMAPLineEdit.setText(str(self.service.evaluatePredictor(config, videoList)['mAP'])),
                               ButtonEnablerManager.setAllButtonsDisabledState(False))).start()





