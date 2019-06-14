import os
import sys
import time
from threading import Thread

import PySide2.QtXml
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QDialog, QLabel
from PySide2.QtCore import QFile, QObject
from tqdm import tqdm

from pedect.config.BasicConfig import BasicConfig, getConfigFromTrainId
from pedect.controller.EvaluatingController import EvaluatingController
from pedect.controller.GenerationController import GenerationController
from pedect.controller.OptimizeController import OptimizeController
from pedect.controller.TrainIdsController import TrainIdsController
from pedect.controller.TrainingController import TrainingController
from pedect.controller.TrainingSetPreparationController import TrainingSetPreparationController
from pedect.service.Service import Service

sys.path.append("./keras-yolo3/")
sys.path.append("./re3-tensorflow/")

class Form(QObject):

    def __init__(self, ui_file, trainIdsController: TrainIdsController, trainingSetPreparationController: TrainingSetPreparationController, trainingController: TrainingController,
                 evaluatingController: EvaluatingController, optimizeController: OptimizeController, generationController: GenerationController, parent=None):
        super(Form, self).__init__(parent)
        ui_file = QFile(ui_file)
        ui_file.open(QFile.ReadOnly)


        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()

        debugConsole = self.window.findChild(QLabel, 'debugConsoleLabel')
        debugConsole.setText("Debug console\n")
        output = StdoutRedirector(debugConsole)
        sys.stdout = output
        sys.stderr = output

        trainIdsController.setUp(self.window)
        trainingSetPreparationController.setUp(self.window)
        trainingController.setUp(self.window)
        evaluatingController.setUp(self.window)
        optimizeController.setUp(self.window)
        generationController.setUp(self.window)

        self.window.show()
        QDialog().show()


realOutput = sys.stdout


class IORedirector(object):
    def __init__(self, label):
        self.label = label


class StdoutRedirector(IORedirector):

    def __init__(self, label):
        IORedirector.__init__(self, label)
        self.lines = []
        self.maxLines = 8

    def write(self, str):
        realOutput.write(str)
        self.lines = self.lines + [x for x in str.split("\n") if x != ""]
        if len(self.lines) > self.maxLines:
            self.lines = self.lines[-self.maxLines:]
        str = ""
        for x in self.lines:
            str += x + "\n"
        self.label.setText(str)

    def flush(self):
        pass




class MyConfig(BasicConfig):
    pass


if __name__ == '__main__':
    app = QApplication(sys.argv)

    MyConfig()
    config = getConfigFromTrainId(11)
    service = Service(config)
    trainingIdsController = TrainIdsController(service)
    trainingPreparationController = TrainingSetPreparationController(service)
    trainController = TrainingController(service, trainingIdsController)
    evaluationController = EvaluatingController(service, trainingIdsController)
    optimizationController = OptimizeController(service, trainingIdsController)
    generatingController = GenerationController(service, trainingIdsController)
    form = Form(os.path.join("pedect", "design", "mainWindow.ui"), trainingIdsController, trainingPreparationController, trainController, evaluationController, optimizationController, generatingController)
    sys.exit(app.exec_())



