import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

_original_qapp_init = QApplication.__init__

def custom_qapp_init(self, *args, **kwargs):
    _original_qapp_init(self, *args, **kwargs)

    self.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, Qt.black)
    palette.setColor(QPalette.Base, Qt.white)
    palette.setColor(QPalette.Text, Qt.black)
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, Qt.black)
    palette.setColor(QPalette.Highlight, QColor(76, 163, 224))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    self.setPalette(palette)

QApplication.__init__ = custom_qapp_init

import sccircuitbuilder
sccircuitbuilder.GUI()
