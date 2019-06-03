import sys

from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QDialog
from PySide2.QtCore import QFile, QObject

from pedect.config.BasicConfig import BasicConfig, getConfigFromTrainId
from pedect.controller.EvaluatingController import EvaluatingController
from pedect.controller.GenerationController import GenerationController
from pedect.controller.OptimizeController import OptimizeController
from pedect.controller.TrainIdsController import TrainIdsController
from pedect.controller.TrainingController import TrainingController
from pedect.controller.TrainingSetPreparationController import TrainingSetPreparationController
from pedect.service.Service import Service

class Form(QObject):

    def __init__(self, ui_file, trainIdsController: TrainIdsController, trainingSetPreparationController: TrainingSetPreparationController, trainingController: TrainingController,
                 evaluatingController: EvaluatingController, optimizeController: OptimizeController, generationController: GenerationController, parent=None):
        super(Form, self).__init__(parent)
        ui_file = QFile(ui_file)
        ui_file.open(QFile.ReadOnly)


        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()
        trainIdsController.setUp(self.window)
        trainingSetPreparationController.setUp(self.window)
        trainingController.setUp(self.window)
        evaluatingController.setUp(self.window)
        optimizeController.setUp(self.window)
        generationController.setUp(self.window)

        self.window.show()
        QDialog().show()


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
    form = Form('pedect/design/mainWindow.ui', trainingIdsController, trainingPreparationController, trainController, evaluationController, optimizationController, generatingController)
    sys.exit(app.exec_())



