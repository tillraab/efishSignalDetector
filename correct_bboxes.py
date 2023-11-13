import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pyqtgraph as pg

from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms.functional as F

from IPython import embed
import pathlib

class ImageWithBbox(pg.GraphicsLayoutWidget):
    def __init__(self, img_path, parent=None):
        super(ImageWithBbox, self).__init__(parent)

        # image setup
        self.imgItem = pg.ImageItem(ColorMap='viridis')
        self.plot_widget = self.addPlot(title="")
        self.plot_widget.addItem(self.imgItem, colorMap='viridis')
        self.plot_widget.scene().sigMouseClicked.connect(self.addROI)

        # self.imgItem.mouseClickEvent()

        self.imgItem.getViewBox().invertY(True)
        self.imgItem.getViewBox().setMouseEnabled(x=False, y=False)

        # get image and labels
        img = Image.open(img_path)
        img_gray = ImageOps.grayscale(img)
        self.img_array = np.array(img_gray).T
        self.imgItem.setImage(np.array(self.img_array))
        self.plot_widget.setYRange(0, self.img_array.shape[0], padding=0)
        self.plot_widget.setXRange(0, self.img_array.shape[1], padding=0)

        label_path = (pathlib.Path(img_path).parent.parent / 'labels' / pathlib.Path(img_path).name).with_suffix('.txt')
        self.labels = np.loadtxt(label_path, delimiter=' ')

        # add ROIS
        self.ROIs = []
        for enu, l in enumerate(self.labels):
            x_center, y_center, width, height = l[1:] * self.img_array.shape[1]
            x0, y0, = x_center-width/2, y_center-height/2
            ROI = pg.RectROI((x0, y0), size=(width, height), removable=True, sideScalers=True)
            ROI.sigRemoveRequested.connect(self.removeROI)

            self.ROIs.append(ROI)
            self.plot_widget.addItem(ROI)

    def removeROI(self, roi):
        if roi in self.ROIs:
            self.ROIs.remove(roi)
            self.plot_widget.removeItem(roi)

    def addROI(self, event):
        # Check if the event is a double-click event
        if event.double():
            pos = event.pos()
            # Transform the mouse position to data coordinates
            pos_data = self.plot_widget.getViewBox().mapToView(pos)
            x, y = pos_data.x(), pos_data.y()
            # Create a new ROI at the double-clicked position
            new_ROI = pg.RectROI(pos=(x, y), size=(self.img_array.shape[0]*0.05, self.img_array.shape[1]*0.05), removable=True)
            new_ROI.sigRemoveRequested.connect(self.removeROI)
            self.ROIs.append(new_ROI)
            self.plot_widget.addItem(new_ROI)

class Bbox_correct_UI(QMainWindow):
    def __init__(self, img_path, parent=None):
        super(Bbox_correct_UI, self).__init__(parent)

        rec = QApplication.desktop().screenGeometry()
        height = rec.height()
        width = rec.width()
        self.resize(int(.8 * width), int(.8 * height))
        self.setWindowTitle('efishSignalTracker')  # set window title

        # widget and layout
        self.central_widget = QWidget(self)
        self.central_gridLayout = QGridLayout()
        self.central_widget.setLayout(self.central_gridLayout)
        self.setCentralWidget(self.central_widget)

        self.current_img = ImageWithBbox(img_path, parent=self)

        self.central_gridLayout.addWidget(self.current_img, 0, 0)

        self.add_actions()

    def add_actions(self):
        self.readout_rois = QAction('read', self)
        self.readout_rois.triggered.connect(self.readout_rois_fn)
        self.readout_rois.setShortcut(Qt.Key_Return)

        self.addAction(self.readout_rois)

    def readout_rois_fn(self):
        for roi in self.current_img.ROIs:
            print(roi.pos(), roi.size())


def main_UI():
    app = QApplication(sys.argv)  # create application
    img_path = sys.argv[1]
    w = Bbox_correct_UI(img_path)  # create window
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main_UI()