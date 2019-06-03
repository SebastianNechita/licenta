from PySide2.QtGui import QStandardItemModel, QStandardItem
from PySide2.QtWidgets import QListView, QPushButton

from pedect.service.Service import Service


class TrainIdsController:
    def __init__(self, service: Service):
        self.window = None
        self.service = service
        self.listViewModel = None
        self.listView = None

    def setUp(self, window):
        self.window = window
        self.listView = window.findChild(QListView, 'trainIdListView')
        self.listViewModel = QStandardItemModel(self.listView)
        self.listView.setModel(self.listViewModel)
        self.__refreshTrainIdList()
        addNewTrainIdButton = window.findChild(QPushButton, 'addNewTrainIdButton')
        addNewTrainIdButton.clicked.connect(self.__addNewTrainId)

    def __refreshTrainIdList(self):
        self.listViewModel.clear()
        for trainId in self.service.getAllTrainIds():
            item = QStandardItem(str(trainId))
            item.setEditable(False)
            self.listViewModel.appendRow(item)

    def __addNewTrainId(self):
        trainId = str(max(self.service.getAllTrainIds()) + 1)
        self.service.createNewTrainingConfiguration(trainId)
        self.__refreshTrainIdList()

    def getSelectedTrainId(self):
        l = self.listView.selectedIndexes()
        if len(l) != 1:
            return None
        index = l[0]
        return self.listViewModel.item(index.row(), index.column()).text()