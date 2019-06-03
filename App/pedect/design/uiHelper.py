from PySide2.QtCore import Qt
from PySide2.QtGui import QStandardItem
from PySide2.QtWidgets import QMessageBox


def showError(text):
    messageBox = QMessageBox()
    messageBox.setWindowTitle("Error!")
    messageBox.setIcon(QMessageBox.Critical)
    messageBox.setText(text)
    messageBox.exec()


def showSuccess(text):
    messageBox = QMessageBox()
    messageBox.setWindowTitle("Success!")
    messageBox.setIcon(QMessageBox.Information)
    messageBox.setText(text)
    messageBox.exec()

def deselectAllFromModel(model):
    for i in range(model.rowCount()):
        item = model.item(i, 0)
        if isinstance(item, QStandardItem):
            item.setCheckState(Qt.CheckState(0))

def selectVideosFromModel(model, videoList):
    trainingSet = set(videoList)
    for i in range(model.rowCount()):
        item = model.item(i, 0)
        if isinstance(item, QStandardItem):
            item.setCheckState(Qt.CheckState(0))
            if tuple(item.data(1)) in trainingSet:
                item.setCheckState(Qt.CheckState(2))

def populateModel(model, videoList):
    for tpl in videoList:
        item = QStandardItem(str(tpl))
        item.setCheckable(True)
        item.setData(tpl, 1)
        model.appendRow(item)

def between0And1(number: float):
    assert 0.0 <= number <= 1.0, "%f must be between 0.0 and 1.0 inclusive" % number
    return number

def getCheckedVideos(model):
    trainList = []
    for i in range(model.rowCount()):
        item = model.item(i, 0)
        if isinstance(item, QStandardItem):
            if item.checkState() == Qt.CheckState.Checked:
                trainList.append(tuple(item.data(1)))
    return trainList