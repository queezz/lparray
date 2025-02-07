from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.dockarea import DockArea, Dock
import pyqtgraph as pg
import sys
import time

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
win.resize(300, 300)
area = DockArea()
win.setCentralWidget(area)
d1 = Dock("cmd output", size=(300, 300))
area.addDock(d1)

w1 = pg.LayoutWidget()
gui_cmd_bw = QtGui.QTextBrowser()
w1.addWidget(gui_cmd_bw, 0, 0)

d1.addWidget(w1)


class EmittingStream(QtCore.QObject):

    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


def normalOutputWritten(text):
    """Append text to the QTextEdit."""
    timestamp = "{}> ".format(time.strftime("%X"))
    if len(text) > 1:
        text = timestamp + text
    gui_cmd_bw.insertPlainText(text)


# cmd output to application browser
sys.stdout = EmittingStream(textWritten=normalOutputWritten)
sys.stderr = EmittingStream(textWritten=normalOutputWritten)

win.show()
if __name__ == "__main__":
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        print("stdout text")
        QtGui.QApplication.instance().exec_()
