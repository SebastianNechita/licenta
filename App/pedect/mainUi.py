import os
import sys
import time

from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QDialog, QLabel, QTextEdit
from PySide2.QtCore import QFile, QObject
from threading import Thread
from tqdm import tqdm

sys.path.append("./keras-yolo3/")
sys.path.append("./re3-tensorflow/")
sys.path.append("./pedect/")


class Form(QObject):

    def __init__(self, ui_file, parent=None):
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

        thread1 = Thread(target=fun1, args=(12, 10))

        thread1.start()

        self.window.show()
        QDialog().show()


realOutput = sys.stdout


def fun1(a, b):
    for i in tqdm(range(10000)):
        c = a + b + i
        # print(c)
        time.sleep(0.01)


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


if __name__ == '__main__':
    app = QApplication(sys.argv)

    form = Form(os.path.join(".", "design", "mainWindow.ui"))
    sys.exit(app.exec_())
