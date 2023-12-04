import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pyqtgraph as pg
import shutil

from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F

from IPython import embed
import pathlib

NONE_CHECKED_CHECKER_COLORS = ['red', 'black']

class ImageWithBbox(pg.GraphicsLayoutWidget):
# class ImageWithBbox(pg.ImageView):
    def __init__(self, img_path, parent=None):
        super(ImageWithBbox, self).__init__(parent)
        self.img_path = img_path
        self.label_path = (pathlib.Path(img_path).parent.parent / 'labels' / pathlib.Path(img_path).name).with_suffix('.txt')

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

        # label_path = (pathlib.Path(img_path).parent.parent / 'labels' / pathlib.Path(img_path).name).with_suffix('.txt')
        self.labels = np.loadtxt(self.label_path, delimiter=' ')
        if len(self.labels) > 0 and len(self.labels.shape) == 1:
            self.labels = np.expand_dims(self.labels, 0)

        # add ROIS
        self.ROIs = []
        for enu, l in enumerate(self.labels):
            # x_center, y_center, width, height = l[1:] * self.img_array.shape[1]
            x_center = l[1] * self.img_array.shape[0]
            y_center = l[2] * self.img_array.shape[1]
            width = l[3] * self.img_array.shape[0]
            height = l[4] * self.img_array.shape[1]

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
    def __init__(self, data_path, parent=None):
        super(Bbox_correct_UI, self).__init__(parent)

        self.data_path = data_path
        self.files = sorted(list(pathlib.Path(self.data_path).absolute().rglob('*images/*.png')))

        self.close_without_saving_rois = False

        rec = QApplication.desktop().screenGeometry()
        height = rec.height()
        width = rec.width()
        self.resize(int(.8 * width), int(.8 * height))


        # widget and layout
        self.central_widget = QWidget(self)
        self.central_gridLayout = QGridLayout()
        self.central_Layout = QHBoxLayout()
        self.central_widget.setLayout(self.central_Layout)
        self.setCentralWidget(self.central_widget)

        ###########
        self.highlighted_label = None
        self.all_labels = []
        self.load_or_create_file_dict()

        new_name, new_file, new_checked = self.file_dict.iloc[0].values
        self.setWindowTitle(f'efishSignalTracker | {new_name} | {self.file_dict["checked"].sum()}/{len(self.file_dict)}')

        # image widget
        self.current_img = ImageWithBbox(new_file, parent=self)
        self.central_Layout.addWidget(self.current_img, 4)

        # image select widget
        self.scroll = QScrollArea()             # Scroll Area which contains the widgets, set as the centralWidget
        self.widget = QWidget()                 # Widget that contains the collection of Vertical Box
        self.vbox = QVBoxLayout()

        for i in range(len(self.file_dict)):
            label = QLabel(f'{self.file_dict["name"][i]}')
            # label.setFrameShape(QLabel.Panel)
            # label.setFrameShadow(QLabel.Sunken)
            label.setAlignment(Qt.AlignRight)
            label.mousePressEvent = lambda event, label=label: self.label_clicked(label)
            # label.mousePressEvent = lambda event, label: self.label_clicked(label)

            if i == 0:
                label.setStyleSheet("border: 2px solid black; "
                                    "color : %s;" % (NONE_CHECKED_CHECKER_COLORS[self.file_dict['checked'][i]]))
                self.highlighted_label = label
            else:
                label.setStyleSheet("border: 1px solid gray; "
                                    "color : %s;" % (NONE_CHECKED_CHECKER_COLORS[self.file_dict['checked'][i]]))
            self.vbox.addWidget(label)
            self.all_labels.append(label)

        self.widget.setLayout(self.vbox)

        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)

        self.central_Layout.addWidget(self.scroll, 1)

        self.add_actions()

        self.init_MenuBar()

    def load_or_create_file_dict(self):
        csvs_im_data_path = list(pathlib.Path(self.data_path).absolute().rglob('*file_dict.csv'))
        if len(csvs_im_data_path) == 0:
            self.file_dict = pd.DataFrame(
                {'name': [f.name for f in self.files],
                 'files': self.files,
                 'checked': np.zeros(len(self.files), dtype=int)} # change this to locked
            )
        else:
            self.file_dict = pd.read_csv(csvs_im_data_path[0], sep=',')

    def save_file_dict(self):
        self.file_dict.to_csv(pathlib.Path(self.data_path)/'file_dict.csv', sep=',', index=False)

    def label_clicked(self, clicked_label):
        if self.highlighted_label:
            hl_mask = self.file_dict['name'] == self.highlighted_label.text()
            hl_label_name, _, hl_checked = self.file_dict[hl_mask].values[0]
            self.highlighted_label.setStyleSheet("border: 1px solid gray; "
                                                 "color: %s;" % (NONE_CHECKED_CHECKER_COLORS[hl_checked]))

        mask = self.file_dict['name'] == clicked_label.text()
        new_name, new_file, new_checked = self.file_dict[mask].values[0]
        clicked_label.setStyleSheet("border: 2px solid black;"
                                    "color: %s;" % (NONE_CHECKED_CHECKER_COLORS[new_checked]))
        self.highlighted_label = clicked_label

        self.switch_to_new_file(new_file, new_name)

    def lock_file(self):

        hl_mask = self.file_dict['name'] == self.highlighted_label.text()
        hl_label_name, _, hl_checked = self.file_dict[hl_mask].values[0]

        # ToDo: do everything with the index instead of mask
        df_idx = self.file_dict.loc[self.file_dict['name'] == self.highlighted_label.text()].index.values[0]
        self.file_dict.at[df_idx, 'checked'] = 1

        self.highlighted_label.setStyleSheet("border: 1px solid gray; "
                                             "color: 'black';")

        new_idx = df_idx + 1 if df_idx < len(self.file_dict)-1 else 0
        new_name, new_file, new_checked = self.file_dict.iloc[new_idx].values

        self.all_labels[new_idx].setStyleSheet("border: 2px solid black;" 
                                               "color: %s;" % (NONE_CHECKED_CHECKER_COLORS[new_checked]))
        self.highlighted_label = self.all_labels[new_idx]

        self.switch_to_new_file(new_file, new_name)


    def switch_to_new_file(self, new_file, new_name):
        self.readout_rois()

        self.setWindowTitle(f'efishSignalTracker | {new_name} | {self.file_dict["checked"].sum()}/{len(self.file_dict)}')

        self.central_Layout.removeWidget(self.current_img)
        self.current_img = ImageWithBbox(new_file, parent=self)
        self.central_Layout.insertWidget(0, self.current_img, 4)

    def readout_rois(self):
        if self.close_without_saving_rois:
            return
        new_labels = []
        for roi in self.current_img.ROIs:
            x0, y0 = roi.pos()
            x0 /= self.current_img.img_array.shape[0]
            y0 /= self.current_img.img_array.shape[1]

            w, h = roi.size()
            w /= self.current_img.img_array.shape[0]
            h /= self.current_img.img_array.shape[1]

            x_center = x0 + w/2
            y_center = y0 + h/2
            new_labels.append([1, x_center, y_center, w, h])
        new_labels = np.array(new_labels)
        np.savetxt(self.current_img.label_path, new_labels)

    def export_validated_data(self):
        fd = QFileDialog()
        export_path = fd.getExistingDirectory(self, 'Select Directory')
        if export_path:
            export_idxs = list(self.file_dict['files'][self.file_dict['checked'] == 1].index)
            keep_idxs = []
            for export_file_path, export_idx in zip(list(self.file_dict['files'][self.file_dict['checked'] == 1]), export_idxs):
                export_image_path = pathlib.Path(export_file_path)
                export_label_path = export_image_path.parent.parent / 'labels' / pathlib.Path(export_image_path.name).with_suffix('.txt')

                target_image_path = pathlib.Path(export_path) / 'images' / export_image_path.name
                target_label_path = pathlib.Path(export_path) / 'labels' / export_label_path.name
                if not target_image_path.exists():
                    # ToDo: this is not tested but should work
                    # os.rename(export_image_path, target_image_path)
                    shutil.copy(export_image_path, target_image_path)
                    # os.rename(export_label_path, target_label_path)
                    shutil.copy(export_label_path, target_label_path)
                else:
                    keep_idxs.append(export_idx)


            # self.file_dict.loc[self.file_dict['name'] == self.highlighted_label.text()].index.values[0]
            drop_idxs = list(set(export_idxs) - set(keep_idxs))
            # self.file_dict = self.file_dict.drop(drop_idxs)
            self.save_file_dict()
            self.close_without_saving_rois = True
            self.close()

        else:
            print('nope')
            pass

    def open(self):
        pass

    def init_MenuBar(self):
        menubar = self.menuBar() # needs QMainWindow ?!
        file = menubar.addMenu('&File') # create file menu ... accessable with alt+F
        file.addActions([self.Act_open, self.Act_export, self.Act_exit])

        edit = menubar.addMenu('&Help')
        # edit.addActions([self.Act_undo, self.Act_unassigned_funds])

    def add_actions(self):
        self.lock_file_act = QAction('loc', self)
        self.lock_file_act.triggered.connect(self.lock_file)
        self.lock_file_act.setShortcut(Qt.Key_Space)
        self.addAction(self.lock_file_act)

        self.Act_open = QAction('&Open', self)
        # self.Act_open.setStatusTip('Open file')
        self.Act_open.triggered.connect(self.open) # ToDo: implement this fn

        self.Act_export = QAction('&Export', self)
        # self.Act_export.setStatusTip('Open file')
        self.Act_export.triggered.connect(self.export_validated_data)

        self.Act_exit = QAction('&Exit', self)  # trigger with alt+E
        self.Act_exit.setShortcut(Qt.Key_Q)
        self.Act_exit.triggered.connect(self.close)

    def closeEvent(self, *args, **kwargs):
        super(QMainWindow, self).closeEvent(*args, **kwargs)
        self.readout_rois()
        self.save_file_dict()

def main_UI():
    app = QApplication(sys.argv)  # create application
    data_path = sys.argv[1]
    w = Bbox_correct_UI(data_path)  # create window
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main_UI()