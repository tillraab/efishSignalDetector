import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pyqtgraph as pg

from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms.functional as F

import pathlib

class Bbox_correct_UI(QMainWindow):
    def __init__(self, img_path, parent=None):
        super(Bbox_correct_UI, self).__init__(parent)
        label_path = (pathlib.Path(img_path).parent.parent / 'labels' / pathlib.Path(img_path).name).with_suffix('.txt')

        rec = QApplication.desktop().screenGeometry()
        height = rec.height()
        width = rec.width()
        self.resize(int(.8 * width), int(.8 * height))
        self.setWindowTitle('efishSignalTracker')  # set window title

        self.central_widget = QWidget(self)
        self.gridLayout = QGridLayout()
        self.central_widget.setLayout(self.gridLayout)
        self.setCentralWidget(self.central_widget)

        self.plot_handels = []
        self.plot_widgets = []
        self.win = pg.GraphicsLayoutWidget()


        self.plot_handels.append(pg.ImageItem(ColorMap='viridis'))
        self.plot_widgets.append(self.win.addPlot(title=""))
        # xxx.setMouseMode(xxx.RectMode)

        self.plot_widgets[0].addItem(self.plot_handels[0], colorMap='viridis')
        # self.plot_widgets[0].setLabel('left', 'frequency [Hz]')
        # self.plot_widgets[0].setLabel('bottom', 'time [s]')

        self.gridLayout.addWidget(self.win, 0, 0)

        self.plot_handels[0].getViewBox().invertY(True)
        self.plot_handels[0].getViewBox().setMouseEnabled(x=False, y=False)
        img = Image.open(img_path)
        img_gray = ImageOps.grayscale(img)
        img_array = np.array(img_gray).T


        self.labels = np.loadtxt(label_path, delimiter=' ')

        self.plot_handels[0].setImage(np.array(img_array))

        self.ROIs = []
        for enu, l in enumerate(self.labels):
            x_center, y_center, width, height = l[1:] * img_array.shape[1]
            x0, y0, = x_center-width/2, y_center-height/2
            ROI = pg.RectROI((x0, y0), size=(width, height), removable=True)
            # ROI.sigRemoveRequested.connect(lambda: )
            self.ROIs.append(ROI)
            self.plot_widgets[0].addItem(ROI)


        self.plot_widgets[0].setYRange(0, img_array.shape[0], padding=0)
        self.plot_widgets[0].setXRange(0, img_array.shape[1], padding=0)

    # def kill_me(self, ROI):
    #     print('yay')
    #     print(ROI)
    #     self.win.removeItem(ROI)

def main_UI():
    app = QApplication(sys.argv)  # create application
    img_path = sys.argv[1]
    w = Bbox_correct_UI(img_path)  # create window
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main_UI()