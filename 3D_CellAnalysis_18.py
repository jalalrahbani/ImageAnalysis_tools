#!/usr/bin/env python3
# This script runs with Python 3.10.0
# Part of the 3D Cell Analysis project
# -*- coding: utf-8 -*-

import sys, os, cv2, numpy as np, tifffile, vtk
from scipy.ndimage import binary_closing, binary_fill_holes
from sklearn.ensemble import RandomForestClassifier

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QToolButton,
    QLabel, QPushButton, QAction, QFileDialog, QSlider, QStatusBar,
    QSpacerItem, QSizePolicy, QMessageBox, QStackedWidget, QDialog, QDialogButtonBox, QMenu,
    QProgressDialog
)
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QPolygonF, QColor
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

###############################################################################
# Helper Function: Convert image to grayscale
###############################################################################
def to_grayscale(image):
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        if image.shape[2] in (3,4):
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            return image
    else:
        return image

###############################################################################
# Freeform ROI selection widget (for background correction and deletion)
###############################################################################
class FreeformImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.freeform_mode = False
        self.drawing = False
        self.points = []
        self.freeform_polygon = None
        self.roi_selected_callback = None
        self.update_pixel_callback = None

    def start_freeform(self):
        self.freeform_mode = True
        self.drawing = False
        self.points = []
        self.freeform_polygon = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.freeform_mode and self.points:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            polygon = QPolygonF(self.points)
            painter.drawPolygon(polygon)

    def mousePressEvent(self, event):
        if self.freeform_mode:
            self.points = [event.pos()]
            self.drawing = True
            self.update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.freeform_mode and self.drawing:
            self.points.append(event.pos())
            self.update()
        if self.update_pixel_callback:
            self.update_pixel_callback(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.freeform_mode and self.drawing:
            self.points.append(event.pos())
            self.drawing = False
            self.freeform_polygon = QPolygonF(self.points)
            self.freeform_mode = False
            self.update()
            if self.roi_selected_callback:
                self.roi_selected_callback(self.freeform_polygon)
        else:
            super().mouseReleaseEvent(event)

###############################################################################
# Brush Image label for ML pixel classification
###############################################################################
class BrushImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.drawing = False
        self.brush_size = 10
        # mode: 1 for signal (green), 2 for background (magenta)
        self.mode = 1
        self.annotation_mask = None
        self.image = None

    def setImage(self, qimage):
        self.image = qimage.copy()
        self.annotation_mask = np.zeros((qimage.height(), qimage.width()), dtype=np.uint8)
        self.setPixmap(QPixmap.fromImage(self.image))
    
    def setBrushSize(self, size):
        self.brush_size = size

    def setMode(self, mode):
        self.mode = mode

    def mousePressEvent(self, event):
        self.drawing = True
        self.drawBrush(event.pos())
    
    def mouseMoveEvent(self, event):
        if self.drawing:
            self.drawBrush(event.pos())
    
    def mouseReleaseEvent(self, event):
        self.drawing = False

    def drawBrush(self, pos):
        if self.image is None or self.annotation_mask is None:
            return
        painter = QPainter(self.image)
        pen_color = QColor(0, 255, 0) if self.mode == 1 else QColor(255, 0, 255)
        painter.setPen(QPen(pen_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPoint(pos)
        painter.end()
        x, y = pos.x(), pos.y()
        cv2.circle(self.annotation_mask, (x, y), self.brush_size // 2, self.mode, -1)
        self.setPixmap(QPixmap.fromImage(self.image))

###############################################################################
# Threshold ROI Dialog
###############################################################################
class ThresholdROIDialog(QDialog):
    def __init__(self, image_volume, initial_frame=0, initial_roi_results=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Threshold ROI Selector")
        if image_volume.ndim == 2:
            self.image_volume = image_volume
            self.num_frames = 1
        elif image_volume.ndim == 3:
            if image_volume.shape[2] in (3,4):
                self.image_volume = cv2.cvtColor(image_volume, cv2.COLOR_RGB2GRAY)
                self.num_frames = 1
            else:
                self.image_volume = image_volume
                self.num_frames = image_volume.shape[0]
        elif image_volume.ndim == 4:
            num_frames = image_volume.shape[0]
            grayscale_frames = []
            for i in range(num_frames):
                frame = image_volume[i]
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.shape[2] in (3,4) else frame
                grayscale_frames.append(gray)
            self.image_volume = np.array(grayscale_frames)
            self.num_frames = self.image_volume.shape[0]
        else:
            self.image_volume = image_volume
            self.num_frames = 1

        self.current_frame = initial_frame
        self.roi_results = initial_roi_results.copy() if initial_roi_results else {}

        layout = QVBoxLayout(self)
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(300,300)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)
        self.info_label = QLabel(f"Frame: {self.current_frame+1} / {self.num_frames}")
        layout.addWidget(self.info_label)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.update_preview)
        layout.addWidget(self.threshold_slider)
        btn_layout = QHBoxLayout()
        self.back_btn = QPushButton("Back 10")
        self.back_btn.clicked.connect(self.go_back)
        btn_layout.addWidget(self.back_btn)
        self.forward_btn = QPushButton("Forward 10")
        self.forward_btn.clicked.connect(self.go_forward)
        btn_layout.addWidget(self.forward_btn)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_current_roi)
        btn_layout.addWidget(self.apply_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_rois)
        btn_layout.addWidget(self.clear_btn)
        self.render_btn = QPushButton("Render")
        self.render_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.render_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.update_preview()

    def compute_contour(self):
        frame = self.image_volume if self.num_frames == 1 else self.image_volume[self.current_frame]
        thresh_val = self.threshold_slider.value()
        ret, binary = cv2.threshold(frame, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, binary
        largest = max(contours, key=cv2.contourArea)
        return largest, binary

    def update_preview(self):
        frame = self.image_volume.copy() if self.num_frames == 1 else self.image_volume[self.current_frame].copy()
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        contour, _ = self.compute_contour()
        if contour is not None:
            cv2.drawContours(frame_color, [contour], -1, (0,255,255), 2)
        height, width, _ = frame_color.shape
        bytes_per_line = 3 * width
        qimg = QImage(frame_color.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.info_label.setText(f"Frame: {self.current_frame+1} / {self.num_frames}")

    def go_back(self):
        self.current_frame = max(0, self.current_frame - 10)
        self.update_preview()

    def go_forward(self):
        self.current_frame = min(self.num_frames - 1, self.current_frame + 10)
        self.update_preview()

    def apply_current_roi(self):
        contour, binary = self.compute_contour()
        if contour is None:
            QMessageBox.warning(self, "No ROI", "No contours found for the current threshold.")
            return
        frame = self.image_volume if self.num_frames == 1 else self.image_volume[self.current_frame]
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        self.roi_results[self.current_frame] = {"mask": mask, "contour": contour}
        self.update_preview()

    def clear_rois(self):
        if QMessageBox.question(self, "Clear ROIs", "Are you sure you want to clear all ROIs?",
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.roi_results = {}
            self.update_preview()

###############################################################################
# MLPixelClassifierDialog – for training on a slice
###############################################################################
class MLPixelClassifierDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ML Pixel Classifier")
        self.image = image
        self.brush_label = BrushImageLabel(self)
        h, w = image.shape
        qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8).convertToFormat(QImage.Format_RGB32)
        self.brush_label.setImage(qimg)
        self.signal_button = QPushButton("Signal")
        self.signal_button.clicked.connect(lambda: self.brush_label.setMode(1))
        self.background_button = QPushButton("Background")
        self.background_button.clicked.connect(lambda: self.brush_label.setMode(2))
        self.train_button = QPushButton("Train & Apply")
        self.train_button.clicked.connect(self.train_and_apply)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(50)
        self.brush_slider.setValue(10)
        self.brush_slider.valueChanged.connect(lambda v: self.brush_label.setBrushSize(v))
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.brush_label)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.signal_button)
        hlayout.addWidget(self.background_button)
        hlayout.addWidget(self.train_button)
        hlayout.addWidget(self.cancel_button)
        layout.addLayout(hlayout)
        layout.addWidget(QLabel("Brush Size"))
        layout.addWidget(self.brush_slider)
        
        self.result_segmentation = None
        # These attributes will be set during training:
        self.training_data = None
        self.training_labels = None

    def train_and_apply(self):
        ann = self.brush_label.annotation_mask
        sig_idx = np.where(ann == 1)
        bg_idx = np.where(ann == 2)
        if len(sig_idx[0]) == 0 or len(bg_idx[0]) == 0:
            QMessageBox.warning(self, "Insufficient Data", "Please annotate both signal and background.")
            return
        X_train = []
        y_train = []
        for i, j in zip(*sig_idx):
            X_train.append([float(self.image[i, j])])
            y_train.append(1)
        for i, j in zip(*bg_idx):
            X_train.append([float(self.image[i, j])])
            y_train.append(0)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X_train, y_train)
        self.trained_classifier = clf
        self.training_data = X_train
        self.training_labels = y_train
        h, w = self.image.shape
        X_all = self.image.reshape(-1, 1).astype(np.float32)
        y_pred = clf.predict(X_all).reshape(h, w)
        seg_color = np.zeros((h, w, 3), dtype=np.uint8)
        seg_color[y_pred == 1] = [0, 255, 0]
        seg_color[y_pred == 0] = [255, 0, 255]
        self.result_segmentation = seg_color
        qimg_seg = QImage(seg_color.data, w, h, 3*w, QImage.Format_RGB888)
        self.brush_label.setPixmap(QPixmap.fromImage(qimg_seg))
        self.accept()

###############################################################################
# FilterSurfacesDialog – for filtering surfaces by volume (with progress)
###############################################################################
class FilterSurfacesDialog(QDialog):
    def __init__(self, polydata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Surfaces by Size")
        self.polydata = polydata
        self.lower_slider = QSlider(Qt.Horizontal)
        self.lower_slider.setMinimum(0)
        self.lower_slider.setMaximum(1000)
        self.lower_slider.setValue(0)
        self.lower_slider.valueChanged.connect(self.update_preview)
        self.upper_slider = QSlider(Qt.Horizontal)
        self.upper_slider.setMinimum(0)
        self.upper_slider.setMaximum(1000)
        self.upper_slider.setValue(1000)
        self.upper_slider.valueChanged.connect(self.update_preview)
        self.lower_label = QLabel("Lower: 0")
        self.upper_label = QLabel("Upper: 1000")
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.apply_filter)
        button_box.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.lower_label)
        slider_layout.addWidget(self.lower_slider)
        slider_layout.addWidget(self.upper_label)
        slider_layout.addWidget(self.upper_slider)
        layout.addLayout(slider_layout)
        layout.addWidget(self.vtk_widget)
        layout.addWidget(button_box)
        self.filtered_polydata = None
        self.update_preview()

    def update_preview(self):
        if self.parent() is not None and hasattr(self.parent(), "set_status"):
            self.parent().set_status("Processing ...")
        lower = self.lower_slider.value()
        upper = self.upper_slider.value()
        self.lower_label.setText(f"Lower: {lower}")
        self.upper_label.setText(f"Upper: {upper}")
        # Use a progress dialog to iterate over regions.
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputData(self.polydata)
        connectivity.SetExtractionModeToAllRegions()
        connectivity.Update()
        num_regions = connectivity.GetNumberOfExtractedRegions()
        prog = QProgressDialog("Filtering surfaces...", "Cancel", 0, num_regions, self)
        prog.setWindowTitle("Filtering")
        prog.setWindowModality(Qt.WindowModal)
        appendFilter = vtk.vtkAppendPolyData()
        for regionId in range(num_regions):
            prog.setValue(regionId)
            QApplication.processEvents()
            if prog.wasCanceled():
                break
            regionConnectivity = vtk.vtkPolyDataConnectivityFilter()
            regionConnectivity.SetInputData(self.polydata)
            regionConnectivity.SetExtractionModeToSpecifiedRegions()
            regionConnectivity.AddSpecifiedRegion(regionId)
            regionConnectivity.Update()
            regionPoly = regionConnectivity.GetOutput()
            massProps = vtk.vtkMassProperties()
            massProps.SetInputData(regionPoly)
            massProps.Update()
            volume = massProps.GetVolume()
            if lower <= volume <= upper:
                appendFilter.AddInputData(regionPoly)
        prog.setValue(num_regions)
        appendFilter.Update()
        filtered_polydata = appendFilter.GetOutput()
        self.vtk_renderer.RemoveAllViewProps()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(filtered_polydata)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        if self.parent() is not None and hasattr(self.parent(), "set_status"):
            self.parent().set_status("Ready", 2000)
        self.filtered_polydata = filtered_polydata

    def apply_filter(self):
        if self.parent() is not None and hasattr(self.parent(), "set_status"):
            self.parent().set_status("Processing ...")
        # (Re-run the update_preview filtering)
        self.update_preview()
        if self.parent() is not None and hasattr(self.parent(), "set_status"):
            self.parent().set_status("Ready", 2000)
        self.accept()


###############################################################################
# Main Application
###############################################################################
class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Cell Analysis")
        self.resize(1000, 700)
        # Data holders
        self.raw_image_data = None
        self.corrected_image_data = None
        self.adjusted_image_data = None
        self.roi_results = {}
        self.num_frames = 1
        self.current_frame_index = 0
        self.brightness = 0
        self.contrast = 1.0
        self.current_view_mode = "2D"
        self.show_raw = True
        self.process_log = []
        self.current_polydata = None
        self.final_polydata = None
        self.selected_surface = None
        # For ML training
        self.ml_X_train = None
        self.ml_y_train = None
        self.ml_training_data = {}
        self.use_gpu = False
        self.init_ui()
        self.gpu_supported = self.is_gpu_supported()
        self.create_menu()
        self.setStatusBar(QStatusBar(self))
        self.set_status("Ready")

    def is_gpu_supported(self):
        try:
            gpu_mapper = vtk.vtkGPUVolumeRayCastMapper()
            # IsRenderSupported returns 1 if GPU volume rendering is supported on this render window.
            if gpu_mapper.IsRenderSupported(self.vtk_widget.GetRenderWindow()) == 1:
                return True
            return False
        except Exception as e:
            return False

    def toggle_gpu_acceleration(self, checked):
        if checked:
            if self.is_gpu_supported():
                self.use_gpu = True
                QMessageBox.information(self, "GPU Acceleration", "GPU Acceleration enabled.")
            else:
                self.use_gpu = False
                QMessageBox.warning(self, "GPU Acceleration", "GPU Acceleration is not supported on your system.")
        else:
            self.use_gpu = False
            QMessageBox.information(self, "GPU Acceleration", "GPU Acceleration disabled.")

    def set_status(self, message, timeout=0):
        gpu_indicator = "GPU: ●" if self.gpu_supported else "GPU: X"
        self.statusBar().showMessage(f"{gpu_indicator} | {message}", timeout)

    # New method: Close current image (clear all loaded data)
    def close_image(self):
        self.raw_image_data = None
        self.corrected_image_data = None
        self.adjusted_image_data = None
        self.roi_results = {}
        self.ml_training_data = {}
        self.ml_X_train = None
        self.ml_y_train = None
        self.current_polydata = None
        self.final_polydata = None
        self.image_label.clear()
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_widget.GetRenderWindow().Render()
        self.z_slider.hide()
        self.z_frame_label.setText("Frame: 0/0")
        self.add_log("Image closed.")

    def init_ui(self):
        top_layout = QHBoxLayout()
        top_layout.addStretch()
        self.pixel_value_label = QLabel("Pixel: N/A")
        self.pixel_value_label.setStyleSheet("background-color: #eee; padding: 2px;")
        top_layout.addWidget(self.pixel_value_label, alignment=Qt.AlignRight)
        self.toggle_button = QPushButton("Switch to 3D View")
        self.toggle_button.clicked.connect(self.toggle_view)
        top_layout.addWidget(self.toggle_button)
        self.toggle_raw_button = QPushButton("Show Corrected")
        self.toggle_raw_button.clicked.connect(self.toggle_raw_corrected)
        top_layout.addWidget(self.toggle_raw_button)
        self.bg_correction_button = QPushButton("Background Correction")
        self.bg_correction_button.clicked.connect(self.start_background_correction)
        top_layout.addWidget(self.bg_correction_button)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.start_delete)
        top_layout.addWidget(self.delete_button)
        self.threshold_roi_button = QPushButton("Threshold ROI")
        self.threshold_roi_button.clicked.connect(self.on_threshold_roi)
        top_layout.addWidget(self.threshold_roi_button)
        self.ml_classifier_button = QPushButton("ML Pixel Classifier")
        self.ml_classifier_button.clicked.connect(self.ml_pixel_classifier)
        top_layout.addWidget(self.ml_classifier_button)
        # ML stack button with dropdown
        self.ml_stack_button = QToolButton()
        self.ml_stack_button.setText("Classify Entire Stack")
        self.ml_stack_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.ml_stack_menu = QMenu(self)
        self.ml_stack_button.setMenu(self.ml_stack_menu)
        combined_action = QAction("Combined", self)
        combined_action.triggered.connect(self.ml_classify_stack_combined)
        self.ml_stack_menu.addAction(combined_action)
        self.ml_slice_menu = QMenu("Slice", self)
        self.ml_stack_menu.addMenu(self.ml_slice_menu)
        top_layout.addWidget(self.ml_stack_button)
        self.filter_surfaces_button = QPushButton("Filter Surfaces")
        self.filter_surfaces_button.clicked.connect(self.filter_surfaces)
        top_layout.addWidget(self.filter_surfaces_button)
        self.delete_surfaces_button = QPushButton("Delete Surfaces")
        self.delete_surfaces_button.clicked.connect(self.delete_surfaces)
        top_layout.addWidget(self.delete_surfaces_button)
        self.surface_menu = QMenu("3D Options", self)
        self.surface_menu.addAction("3D Surface", self.render_roi_surface)
        self.surface_menu.addAction("Final Structure", self.render_final_structure)
        self.surface_button = QPushButton("3D Options")
        self.surface_button.setMenu(self.surface_menu)
        top_layout.addWidget(self.surface_button)
        # Creat a new layout for the second row of buttons
        surface_select_layout = QHBoxLayout()
        surface_select_button = QPushButton("Surface Select")
        surface_select_button.clicked.connect(self.start_surface_selection)
        surface_select_layout.addWidget(surface_select_button)

        # Add the new layout to the central 2D/3D view layout
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.addLayout(top_layout)
        self.display_stack = QStackedWidget()
        self.image_label = FreeformImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222222; color: white;")
        self.image_label.update_pixel_callback = self.handle_mouse_move
        self.display_stack.addWidget(self.image_label)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self.display_stack.addWidget(self.vtk_widget)
        self.display_stack.setCurrentIndex(0)
        central_layout.addWidget(self.display_stack, stretch=1)
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z-Frame:"))
        self.z_frame_label = QLabel("Frame: 0/0")
        z_layout.addWidget(self.z_frame_label)
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.valueChanged.connect(self.change_frame)
        self.z_slider.hide()
        z_layout.addWidget(self.z_slider)
        central_layout.addLayout(z_layout)
        # add the surface selection layout to the central layout
        central_layout.addLayout(surface_select_layout)

        adjust_layout = QHBoxLayout()
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickInterval(10)
        self.brightness_slider.valueChanged.connect(self.adjust_image)
        adjust_layout.addWidget(QLabel("Brightness"))
        adjust_layout.addWidget(self.brightness_slider)
        adjust_layout.addSpacerItem(QSpacerItem(20,20,QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(10)
        self.contrast_slider.setMaximum(300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setTickInterval(10)
        self.contrast_slider.valueChanged.connect(self.adjust_image)
        adjust_layout.addWidget(QLabel("Contrast"))
        adjust_layout.addWidget(self.contrast_slider)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_adjustments)
        adjust_layout.addWidget(self.reset_button)
        self.auto_adjust_button = QPushButton("Auto Adjust")
        self.auto_adjust_button.clicked.connect(self.auto_adjust_image)
        adjust_layout.addWidget(self.auto_adjust_button)
        central_layout.addLayout(adjust_layout)
        self.setCentralWidget(central_widget)

    def create_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open...", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        close_action = QAction("Close Image", self)
        close_action.triggered.connect(self.close_image)
        file_menu.addAction(close_action)
        file_menu.addSeparator()
        save_action = QAction("Save As...", self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        save_log_action = QAction("Save Log", self)
        save_log_action.triggered.connect(self.save_log)
        file_menu.addAction(save_log_action)
        save_rois_action = QAction("Save ROIs", self)
        save_rois_action.triggered.connect(self.save_rois)
        file_menu.addAction(save_rois_action)
        save_3d_surface_action = QAction("Save 3D Surface", self)
        save_3d_surface_action.triggered.connect(self.save_3d_surface)
        file_menu.addAction(save_3d_surface_action)
        save_final_structure_action = QAction("Save Final Structure", self)
        save_final_structure_action.triggered.connect(self.save_final_structure)
        file_menu.addAction(save_final_structure_action)
        file_menu.addSeparator()
        # Add the GPU acceleration toggle action in the File menu

        gpu_action = QAction("Use GPU Acceleration", self)
        gpu_action.setCheckable(True)
        gpu_action.triggered.connect(self.toggle_gpu_acceleration)
        file_menu.addAction(gpu_action)
        file_menu.addSeparator()
        # Add the Open ROIs, 3D Surface, and Final Structure actions
        open_rois_action = QAction("Open ROIs", self)
        open_rois_action.triggered.connect(self.open_rois)
        file_menu.addAction(open_rois_action)
        open_3d_surface_action = QAction("Open 3D Surface", self)
        open_3d_surface_action.triggered.connect(self.open_3d_surface)
        file_menu.addAction(open_3d_surface_action)
        open_final_structure_action = QAction("Open Final Structure", self)
        open_final_structure_action.triggered.connect(self.open_final_structure)
        file_menu.addAction(open_final_structure_action)

    def add_log(self, message):
        self.process_log.append(message)
        print("LOG:", message)

    def on_key_press_surface_selection(self, caller, event):
        key = caller.GetKeySym()
        if key == "Shift_L" or key == "Shift_R":
            self.is_shift_pressed = True  # Enable highlighting

    def on_key_release_surface_selection(self, caller, event):
        key = caller.GetKeySym()
        if key == "Shift_L" or key == "Shift_R":
            self.is_shift_pressed = False  # Disable highlighting

    def start_surface_selection(self):
        # Enable interactor to detect mouse movement
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Create sphere cursor
        self.sphere_cursor = vtk.vtkSphereSource()
        self.sphere_cursor.SetRadius(5.0)

        # Mapper and Actor for cursor visualization
        self.sphere_mapper = vtk.vtkPolyDataMapper()
        self.sphere_mapper.SetInputConnection(self.sphere_cursor.GetOutputPort())

        self.sphere_actor = vtk.vtkActor()
        self.sphere_actor.SetMapper(self.sphere_mapper)
        self.sphere_actor.GetProperty().SetColor(0.5, 0.5, 0.5)  # Grey cursor initially
        self.sphere_actor.GetProperty().SetOpacity(0.5)

        # Add the cursor to the renderer
        self.vtk_renderer.AddActor(self.sphere_actor)
        self.vtk_widget.GetRenderWindow().Render()

        # Add observers for mouse movement and shift key press
        interactor.AddObserver("MouseMoveEvent", self.on_mouse_move_surface_selection)
        interactor.AddObserver("KeyPressEvent", self.on_key_press_surface_selection)
        interactor.AddObserver("KeyReleaseEvent", self.on_key_release_surface_selection)
        self.is_shift_pressed = False  # Track shift key state

    def on_mouse_move_surface_selection(self, caller, event):
        # Get mouse position in world coordinates
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        pos = interactor.GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(pos[0], pos[1], 0, self.vtk_renderer)

        if picker.GetActor():
            world_pos = picker.GetPickPosition()
            self.sphere_cursor.SetCenter(world_pos)
            self.sphere_cursor.Modified()

            # If Shift is held, highlight the surface under the cursor
            if self.is_shift_pressed and self.current_polydata:
                threshold = vtk.vtkThreshold()
                threshold.SetInputData(self.current_polydata)
                threshold.ThresholdBetween(0, 100)  # Modify as needed for selection

                # Extract selected surface
                extractor = vtk.vtkGeometryFilter()
                extractor.SetInputConnection(threshold.GetOutputPort())
                extractor.Update()
                selected_surface = extractor.GetOutput()

                # Update color to fluorescent green
                if selected_surface.GetNumberOfCells() > 0:
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(selected_surface)
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(0, 1, 0)  # Fluorescent green

                    self.vtk_renderer.AddActor(actor)
                    self.vtk_widget.GetRenderWindow().Render()

        return 1

    def on_surface_selection_end(self, caller, event):
        # Get the sphere widget's current parameters.
        center = [0.0, 0.0, 0.0]
        caller.GetCenter(center)
        radius = caller.GetRadius()
        
        # Create an implicit sphere function using these parameters.
        implicit_sphere = vtk.vtkSphere()
        implicit_sphere.SetCenter(center)
        implicit_sphere.SetRadius(radius)
        
        # Use vtkExtractGeometry to extract cells from current_polydata that fall within the sphere.
        extractor = vtk.vtkExtractGeometry()
        extractor.SetInputData(self.current_polydata)
        extractor.SetImplicitFunction(implicit_sphere)
        extractor.Update()
        selected_polydata = extractor.GetOutput()
        
        # Store the selected surfaces for later deletion.
        self.selected_surfaces = selected_polydata
        
        # Highlight the selected surfaces with a dark red color.
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(selected_polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.5, 0, 0)  # Dark red
        # Optionally, add this actor to the renderer (for example, as an overlay).
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        
        # Optionally, turn the sphere widget off.
        caller.Off()


    # --- File operations ---
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not file_path:
            return

        # Clear existing 3D polydata when opening a new image
        self.current_polydata = None
        self.final_polydata = None
        self.vtk_renderer.RemoveAllViewProps()  # Clears any previously rendered surfaces
        self.vtk_widget.GetRenderWindow().Render()

        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in ['.tif', '.tiff']:
                img = tifffile.imread(file_path)
            else:
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError("cv2 failed to load the image.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
            return

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if img.ndim == 2:
            self.num_frames = 1
        elif img.ndim == 3:
            self.num_frames = 1 if img.shape[2] in (3, 4) else img.shape[0]
        elif img.ndim == 4:
            self.num_frames = img.shape[0]
        else:
            self.num_frames = 1

        self.raw_image_data = img.copy()
        self.corrected_image_data = img.copy()
        self.adjusted_image_data = None
        self.current_frame_index = 0

        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.brightness = 0
        self.contrast = 1.0

        if self.num_frames > 1:
            self.z_slider.setMinimum(0)
            self.z_slider.setMaximum(self.num_frames - 1)
            self.z_slider.setValue(0)
            self.z_slider.show()
            self.z_frame_label.setText(f"Frame: 1/{self.num_frames}")
        else:
            self.z_slider.hide()
            self.z_frame_label.setText("Frame: 1/1")

        self.add_log(f"Opened image: {file_path}")
        self.adjust_image()

        if self.current_view_mode == "3D":
            self.update_3d_view()


    def save_image(self):
        if self.adjusted_image_data is None:
            QMessageBox.warning(self, "No image", "No image available to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image As", "",
                                                   "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap (*.bmp);;TIFF Image (*.tif *.tiff)")
        if not file_path:
            return
        img_to_save = self.adjusted_image_data.copy()
        if img_to_save.ndim == 3:
            if img_to_save.shape[2] == 3:
                img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
            elif img_to_save.shape[2] == 4:
                img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGBA2BGRA)
        success = cv2.imwrite(file_path, img_to_save)
        if not success:
            QMessageBox.critical(self, "Error", "Failed to save image.")
        else:
            QMessageBox.information(self, "Saved", f"Image saved to {file_path}")
            self.add_log(f"Saved image: {file_path}")

    def save_log(self):
        if not self.process_log:
            QMessageBox.information(self, "Log", "No log entries available.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Log", "", "Text Files (*.txt)")
        if not file_path:
            return
        try:
            with open(file_path, "w") as f:
                for entry in self.process_log:
                    f.write(entry + "\n")
            QMessageBox.information(self, "Log Saved", f"Log saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save log:\n{str(e)}")

    def save_rois(self):
        if not self.roi_results:
            QMessageBox.warning(self, "No ROI", "No ROI data available to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save ROIs", "", "NumPy Files (*.npy)")
        if not file_path:
            return
        try:
            np.save(file_path, self.roi_results)
            QMessageBox.information(self, "Saved", f"ROIs saved to {file_path}")
            self.add_log(f"Saved ROIs: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save ROIs:\n{str(e)}")

    def save_3d_surface(self):
        if self.current_polydata is None:
            QMessageBox.warning(self, "No Surface", "No 3D surface available to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save 3D Surface", "", "STL Files (*.stl)")
        if not file_path:
            return
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(file_path)
        writer.SetInputData(self.current_polydata)
        writer.Write()
        QMessageBox.information(self, "Saved", f"3D Surface saved to {file_path}")
        self.add_log(f"Saved 3D surface: {file_path}")

    def save_final_structure(self):
        if self.final_polydata is None:
            QMessageBox.warning(self, "No Final Structure", "No final 3D structure available to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Final Structure", "", "STL Files (*.stl)")
        if not file_path:
            return
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(file_path)
        writer.SetInputData(self.final_polydata)
        writer.Write()
        QMessageBox.information(self, "Saved", f"Final structure saved to {file_path}")
        self.add_log(f"Saved final 3D structure: {file_path}")

    def open_rois(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open ROIs", "", "NumPy Files (*.npy)")
        if not file_path:
            return
        try:
            rois = np.load(file_path, allow_pickle=True).item()
            self.roi_results = rois
            QMessageBox.information(self, "Opened", f"ROIs loaded from {file_path}")
            self.add_log(f"Opened ROIs: {file_path}")
            self.update_image_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open ROIs:\n{str(e)}")

    def open_3d_surface(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open 3D Surface", "", "STL Files (*.stl)")
        if not file_path:
            return
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_path)
        reader.Update()
        polydata = reader.GetOutput()
        self.current_polydata = polydata  # store loaded surface
        # Also update ROI results if needed (optional)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.5, 0.5)
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        QMessageBox.information(self, "Opened", f"3D Surface loaded from {file_path}")
        self.add_log(f"Opened 3D surface: {file_path}")

    def open_final_structure(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Final Structure", "", "STL Files (*.stl)")
        if not file_path:
            return
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_path)
        reader.Update()
        polydata = reader.GetOutput()
        self.final_polydata = polydata  # store loaded final structure
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        QMessageBox.information(self, "Opened", f"Final structure loaded from {file_path}")
        self.add_log(f"Opened final 3D structure: {file_path}")

    def on_freeform_roi_selected(self, polygon):
        if self.adjusted_image_data is None:
            return
        if self.adjusted_image_data.ndim == 2:
            orig_h, orig_w = self.adjusted_image_data.shape
        elif self.adjusted_image_data.ndim == 3:
            orig_h, orig_w = (self.adjusted_image_data.shape[:2] if self.adjusted_image_data.shape[2] in (3,4)
                              else self.get_current_frame(self.adjusted_image_data).shape)
        elif self.adjusted_image_data.ndim == 4:
            orig_h, orig_w = self.get_current_frame(self.adjusted_image_data).shape[:2]
        else:
            return
        lw = self.image_label.width()
        lh = self.image_label.height()
        scale = min(lw / orig_w, lh / orig_h)
        disp_w = orig_w * scale
        disp_h = orig_h * scale
        offset_x = (lw - disp_w) / 2
        offset_y = (lh - disp_h) / 2
        poly_points = []
        for pt in polygon:
            x = int((pt.x() - offset_x) / scale)
            y = int((pt.y() - offset_y) / scale)
            x = max(0, min(orig_w - 1, x))
            y = max(0, min(orig_h - 1, y))
            poly_points.append([x, y])
        poly_np = np.array(poly_points, dtype=np.int32)
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_np], 1)
        if self.raw_image_data.ndim == 2:
            background_pixels = self.raw_image_data[mask == 0]
            avg_background = np.mean(background_pixels)
            corrected = np.clip(self.raw_image_data.astype(np.int16) - int(avg_background), 0, 255).astype(np.uint8)
        elif self.raw_image_data.ndim == 3:
            if self.raw_image_data.shape[2] in (3,4):
                gray = cv2.cvtColor(self.raw_image_data, cv2.COLOR_RGB2GRAY)
                background_pixels = gray[mask == 0]
                avg_background = np.mean(background_pixels)
                corrected = np.clip(self.raw_image_data.astype(np.int16) - int(avg_background), 0, 255).astype(np.uint8)
            else:
                num_frames = self.raw_image_data.shape[0]
                corrected = np.empty_like(self.raw_image_data)
                for i in range(num_frames):
                    frame_i = self.raw_image_data[i]
                    background_pixels = frame_i[mask == 0]
                    avg_background = np.mean(background_pixels)
                    corrected[i] = np.clip(frame_i.astype(np.int16) - int(avg_background), 0, 255).astype(np.uint8)
        elif self.raw_image_data.ndim == 4:
            num_frames = self.raw_image_data.shape[0]
            corrected = np.empty_like(self.raw_image_data)
            for i in range(num_frames):
                frame_i = self.raw_image_data[i]
                gray = cv2.cvtColor(frame_i, cv2.COLOR_RGB2GRAY)
                background_pixels = gray[mask == 0]
                avg_background = np.mean(background_pixels)
                corrected[i] = np.clip(frame_i.astype(np.int16) - int(avg_background), 0, 255).astype(np.uint8)
        else:
            return
        QMessageBox.information(self, "Background Correction", "Per-frame background correction applied using the selected ROI.")
        self.add_log("Applied per-frame background subtraction using freeform ROI.")
        self.corrected_image_data = corrected
        self.adjust_image()

    def start_delete(self):
        if self.adjusted_image_data is None:
            QMessageBox.warning(self, "No Image", "No image loaded for deletion.")
            return
        QMessageBox.information(self, "Delete ROI Selection",
            "Please freeform draw around the pixels you want to delete. The selected pixels will be set to 0.")
        self.image_label.start_freeform()
        self.image_label.roi_selected_callback = self.on_delete_roi_selected

    def on_delete_roi_selected(self, polygon):
        if self.raw_image_data is None:
            return
        if self.raw_image_data.ndim == 2:
            orig_h, orig_w = self.raw_image_data.shape
        elif self.raw_image_data.ndim == 3:
            orig_h, orig_w = (self.raw_image_data.shape[:2] if self.raw_image_data.shape[2] in (3,4)
                              else self.get_current_frame(self.raw_image_data).shape)
        elif self.raw_image_data.ndim == 4:
            orig_h, orig_w = self.get_current_frame(self.raw_image_data).shape[:2]
        else:
            return
        lw = self.image_label.width()
        lh = self.image_label.height()
        scale = min(lw / orig_w, lh / orig_h)
        disp_w = orig_w * scale
        disp_h = orig_h * scale
        offset_x = (lw - disp_w) / 2
        offset_y = (lh - disp_h) / 2
        poly_points = []
        for pt in polygon:
            x = int((pt.x() - offset_x) / scale)
            y = int((pt.y() - offset_y) / scale)
            x = max(0, min(orig_w - 1, x))
            y = max(0, min(orig_h - 1, y))
            poly_points.append([x, y])
        poly_np = np.array(poly_points, dtype=np.int32)
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_np], 1)
        if self.raw_image_data.ndim == 2:
            self.raw_image_data[mask == 1] = 0
        elif self.raw_image_data.ndim == 3:
            if self.raw_image_data.shape[2] in (3,4):
                self.raw_image_data[mask == 1] = 0
            else:
                for i in range(self.raw_image_data.shape[0]):
                    self.raw_image_data[i] = np.where(mask == 1, 0, self.raw_image_data[i])
        elif self.raw_image_data.ndim == 4:
            for i in range(self.raw_image_data.shape[0]):
                frame = self.raw_image_data[i]
                mask_expanded = mask[..., np.newaxis]
                self.raw_image_data[i] = np.where(mask_expanded == 1, 0, frame)
        self.corrected_image_data = self.raw_image_data.copy()
        QMessageBox.information(self, "Deletion", "Selected pixels have been deleted.")
        self.add_log("Applied deletion on selected ROI.")
        self.adjust_image()
        
    def toggle_raw_corrected(self):
        self.show_raw = not self.show_raw
        if self.show_raw:
            self.toggle_raw_button.setText("Show Corrected")
            self.add_log("Toggled view: Showing raw image")
        else:
            self.toggle_raw_button.setText("Show Raw")
            self.add_log("Toggled view: Showing background-corrected image")
        self.adjust_image()

    def start_background_correction(self):
        if self.adjusted_image_data is None:
            QMessageBox.warning(self, "No Image", "No image loaded for correction.")
            return
        QMessageBox.information(self, "Freeform ROI Selection",
                                "Please freeform draw around the object of interest. Pixels outside your drawn polygon will be considered background.")
        self.image_label.start_freeform()
        self.image_label.roi_selected_callback = self.on_freeform_roi_selected

    def on_threshold_roi(self):
        if self.current_view_mode != "2D":
            QMessageBox.warning(self, "2D View Required", "Threshold ROI can only be applied in 2D view.")
            return
        base = self.adjusted_image_data if self.adjusted_image_data is not None else self.raw_image_data
        if base is None:
            QMessageBox.warning(self, "No Image", "No image loaded for thresholding.")
            return
        if base.ndim == 2:
            image_vol = base
        elif base.ndim == 3:
            image_vol = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY) if base.shape[2] in (3,4) else base
        elif base.ndim == 4:
            grayscale_frames = []
            for i in range(base.shape[0]):
                frame = base[i]
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.shape[2] in (3,4) else frame
                grayscale_frames.append(gray)
            image_vol = np.array(grayscale_frames)
        else:
            image_vol = base
        dialog = ThresholdROIDialog(image_vol, initial_frame=self.current_frame_index, initial_roi_results=self.roi_results, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.roi_results = dialog.roi_results
            self.add_log("Threshold ROI selection updated.")
            self.update_image_display()

    def ml_pixel_classifier(self):
        base = self.adjusted_image_data
        if base is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        frame = self.get_current_frame(base)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame.copy()
        dialog = MLPixelClassifierDialog(frame_gray, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            if self.ml_X_train is None:
                self.ml_X_train = dialog.training_data
                self.ml_y_train = dialog.training_labels
            else:
                self.ml_X_train = np.concatenate((self.ml_X_train, dialog.training_data), axis=0)
                self.ml_y_train = np.concatenate((self.ml_y_train, dialog.training_labels), axis=0)
            self.ml_training_data[self.current_frame_index] = (dialog.training_data, dialog.training_labels)
            new_clf = RandomForestClassifier(n_estimators=50)
            new_clf.fit(self.ml_X_train, self.ml_y_train)
            self.ml_classifier = new_clf
            mask = (dialog.result_segmentation[:, :, 0] == 0).astype(np.uint8) * 255
            self.roi_results[self.current_frame_index] = {"mask": mask, "contour": None}
            self.update_image_display()
            QMessageBox.information(self, "Segmentation", "ML segmentation applied to current slice.\n(Combined training data is now used.)")
            self.update_ml_slice_menu()

    def ml_classify_slice(self, frame_index):
        if frame_index not in self.ml_training_data:
            QMessageBox.warning(self, "No Training Data", f"No training data available for frame {frame_index}.")
            return
        X_train, y_train = self.ml_training_data[frame_index]
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X_train, y_train)
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        if base is None:
            return
        num_slices = 1 if base.ndim == 2 or (base.ndim == 3 and base.shape[2] in (3,4)) else base.shape[0]
        seg_stack = []
        for i in range(num_slices):
            slice_i = self.get_current_frame(base) if num_slices == 1 else base[i]
            gray = cv2.cvtColor(slice_i, cv2.COLOR_RGB2GRAY) if (slice_i.ndim == 3 and slice_i.shape[2] not in (3,4)) else slice_i.copy()
            h, w = gray.shape
            X_all = gray.reshape(-1, 1).astype(np.float32)
            y_pred = clf.predict(X_all).reshape(h, w)
            seg_color = np.zeros((h, w, 3), dtype=np.uint8)
            seg_color[y_pred == 1] = [0, 255, 0]
            seg_color[y_pred == 0] = [255, 0, 255]
            seg_stack.append(seg_color)
            self.roi_results[i] = {"mask": (y_pred==1).astype(np.uint8)*255, "contour": None}
        self.ml_segmentation_stack = seg_stack
        self.update_image_display()
        QMessageBox.information(self, "Slice Classification", 
            f"ML classification using training from frame {frame_index} applied to entire stack.")

    def ml_classify_stack_combined(self):
        self.set_status("Processing ML classification for entire stack...")
        if not hasattr(self, "ml_classifier"):
            QMessageBox.warning(self, "No Classifier", "Please run the ML Pixel Classifier on the current slice first.")
            return
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        if base is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            self.set_status("Ready", 2000)
            return
        num_slices = 1 if base.ndim == 2 or (base.ndim == 3 and base.shape[2] in (3,4)) else base.shape[0]
        seg_stack = []
        for i in range(num_slices):
            slice_i = self.get_current_frame(base) if num_slices == 1 else base[i]
            gray = cv2.cvtColor(slice_i, cv2.COLOR_RGB2GRAY) if (slice_i.ndim == 3 and slice_i.shape[2] not in (3,4)) else slice_i.copy()
            h, w = gray.shape
            X_all = gray.reshape(-1, 1).astype(np.float32)
            y_pred = self.ml_classifier.predict(X_all).reshape(h, w)
            seg_color = np.zeros((h, w, 3), dtype=np.uint8)
            seg_color[y_pred == 1] = [0, 255, 0]
            seg_color[y_pred == 0] = [255, 0, 255]
            seg_stack.append(seg_color)
            self.roi_results[i] = {"mask": (y_pred==1).astype(np.uint8)*255, "contour": None}
        self.ml_segmentation_stack = seg_stack
        self.update_image_display()
        QMessageBox.information(self, "Stack Classification", "ML classification (combined training) applied to entire stack.")
        self.set_status("Ready", 2000)

    def ml_classify_stack_slice(self, frame_index):
        if frame_index not in self.ml_training_data:
            QMessageBox.warning(self, "No Training Data", f"No training data available for frame {frame_index}.")
            return
        X_train, y_train = self.ml_training_data[frame_index]
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X_train, y_train)
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        if base is None:
            return
        num_slices = 1 if base.ndim == 2 or (base.ndim == 3 and base.shape[2] in (3,4)) else base.shape[0]
        seg_stack = []
        for i in range(num_slices):
            slice_i = self.get_current_frame(base) if num_slices == 1 else base[i]
            gray = cv2.cvtColor(slice_i, cv2.COLOR_RGB2GRAY) if (slice_i.ndim == 3 and slice_i.shape[2] not in (3,4)) else slice_i.copy()
            h, w = gray.shape
            X_all = gray.reshape(-1, 1).astype(np.float32)
            y_pred = clf.predict(X_all).reshape(h, w)
            seg_color = np.zeros((h, w, 3), dtype=np.uint8)
            seg_color[y_pred == 1] = [0, 255, 0]
            seg_color[y_pred == 0] = [255, 0, 255]
            seg_stack.append(seg_color)
            self.roi_results[i] = {"mask": (y_pred==1).astype(np.uint8)*255, "contour": None}
        self.ml_segmentation_stack = seg_stack
        self.update_image_display()
        QMessageBox.information(self, "Slice Classification", 
            f"ML classification using training from frame {frame_index} applied to entire stack.")

    def update_ml_slice_menu(self):
        self.ml_slice_menu.clear()
        for frame in sorted(self.ml_training_data.keys()):
            action = QAction(f"Frame {frame}", self)
            action.triggered.connect(lambda checked, f=frame: self.ml_classify_stack_slice(f))
            self.ml_slice_menu.addAction(action)

    # --- Rendering methods ---
    def render_roi_surface(self):
        self.set_status("Processing ...")
        # If a 3D surface was loaded from file, use that:
        if self.current_polydata is not None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.current_polydata)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.5, 0.5)
            self.vtk_renderer.RemoveAllViewProps()
            self.vtk_renderer.AddActor(actor)
            self.vtk_renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.set_status("Ready", 2000)
            return
        # Otherwise, generate from ROI results
        if not self.roi_results:
            QMessageBox.warning(self, "No ROI", "No ROI outlines available for rendering.")
            return
        if self.raw_image_data is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        if self.raw_image_data.ndim == 2:
            height, width = self.raw_image_data.shape; num_frames = 1
        elif self.raw_image_data.ndim == 3:
            if self.raw_image_data.shape[2] in (3,4):
                height, width = self.raw_image_data.shape[:2]; num_frames = 1
            else:
                num_frames, height, width = self.raw_image_data.shape
        elif self.raw_image_data.ndim == 4:
            num_frames, height, width = self.raw_image_data.shape[0], self.raw_image_data.shape[1], self.raw_image_data.shape[2]
        else:
            return
        vol = np.zeros((num_frames, height, width), dtype=np.uint8)
        for i in range(num_frames):
            if i in self.roi_results:
                vol[i] = self.roi_results[i]["mask"]
        vol = np.ascontiguousarray(vol)
        dims = vol.shape
        importer = vtk.vtkImageImport()
        importer.CopyImportVoidPointer(vol.tobytes(), len(vol.tobytes()))
        importer.SetDataScalarTypeToUnsignedChar()
        importer.SetNumberOfScalarComponents(1)
        importer.SetDataExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
        importer.SetWholeExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
        importer.Update()
        marchingCubes = vtk.vtkMarchingCubes()
        marchingCubes.SetInputConnection(importer.GetOutputPort())
        marchingCubes.SetValue(0, 127)
        marchingCubes.Update()
        num_points = marchingCubes.GetOutput().GetNumberOfPoints()
        if num_points == 0:
            QMessageBox.warning(self, "Thresholding", "No surface was extracted. Try a different threshold value.")
            return
        polyMapper = vtk.vtkPolyDataMapper()
        polyMapper.SetInputConnection(marchingCubes.GetOutputPort())
        polyMapper.ScalarVisibilityOff()
        surfaceActor = vtk.vtkActor()
        surfaceActor.SetMapper(polyMapper)
        surfaceActor.GetProperty().SetColor(1.0, 0.5, 0.5)
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(surfaceActor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.set_status("Ready", 2000)

    def render_final_structure(self):
        self.set_status("Processing ...")
        # If a final structure was loaded from file, use it:
        if self.final_polydata is not None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.final_polydata)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 1.0, 0.0)
            self.vtk_renderer.RemoveAllViewProps()
            self.vtk_renderer.AddActor(actor)
            self.vtk_renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.set_status("Ready", 2000)
            return
        if not self.roi_results:
            QMessageBox.warning(self, "No ROI", "No ROI outlines available for final structure.")
            return
        if self.raw_image_data is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        if self.raw_image_data.ndim == 2:
            height, width = self.raw_image_data.shape; num_frames = 1
        elif self.raw_image_data.ndim == 3:
            if self.raw_image_data.shape[2] in (3,4):
                height, width = self.raw_image_data.shape[:2]; num_frames = 1
            else:
                num_frames, height, width = self.raw_image_data.shape
        elif self.raw_image_data.ndim == 4:
            num_frames, height, width = self.raw_image_data.shape[0], self.raw_image_data.shape[1], self.raw_image_data.shape[2]
        else:
            return
        vol = np.zeros((num_frames, height, width), dtype=np.uint8)
        for i in range(num_frames):
            if i in self.roi_results:
                vol[i] = self.roi_results[i]["mask"]
        vol_bool = vol > 0
        closed = binary_closing(vol_bool, structure=np.ones((3,3,3)))
        filled = binary_fill_holes(closed)
        vol_final = (filled * 255).astype(np.uint8)
        vol_final = np.ascontiguousarray(vol_final)
        dims = vol_final.shape
        importer = vtk.vtkImageImport()
        importer.CopyImportVoidPointer(vol_final.tobytes(), len(vol_final.tobytes()))
        importer.SetDataScalarTypeToUnsignedChar()
        importer.SetNumberOfScalarComponents(1)
        importer.SetDataExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
        importer.SetWholeExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
        importer.Update()
        marchingCubes = vtk.vtkMarchingCubes()
        marchingCubes.SetInputConnection(importer.GetOutputPort())
        marchingCubes.SetValue(0, 127)
        marchingCubes.Update()
        num_points = marchingCubes.GetOutput().GetNumberOfPoints()
        if num_points == 0:
            QMessageBox.warning(self, "Final Structure", "No surface was extracted. Please check your ROI outlines.")
            return
        polyMapper = vtk.vtkPolyDataMapper()
        polyMapper.SetInputConnection(marchingCubes.GetOutputPort())
        polyMapper.ScalarVisibilityOff()
        surfaceActor = vtk.vtkActor()
        surfaceActor.SetMapper(polyMapper)
        surfaceActor.GetProperty().SetColor(1.0, 1.0, 0.0)
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(surfaceActor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.set_status("Ready", 2000)

    # --- Image adjustment and display methods ---
    def adjust_image(self):
        if self.raw_image_data is None:
            return
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        frame = self.get_current_frame(base)
        frame_float = frame.astype(np.float32)
        self.brightness = self.brightness_slider.value()
        self.contrast = self.contrast_slider.value() / 100.0
        adjusted = np.clip(self.contrast * frame_float + self.brightness, 0, 255).astype(np.uint8)
        self.adjusted_image_data = adjusted
        self.update_image_display()
        self.add_log(f"Applied brightness/contrast: brightness={self.brightness}, contrast={self.contrast}")
        if self.current_view_mode == "3D" and self.num_frames > 1:
            self.update_3d_view()

    def update_image_display(self):
        if self.adjusted_image_data is None:
            return
        frame = self.get_current_frame(self.adjusted_image_data).copy()
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        if self.roi_results and self.current_frame_index in self.roi_results:
            contour = self.roi_results[self.current_frame_index]["contour"]
            cv2.drawContours(frame_color, [contour], -1, (0,255,255), 2)
        frame_rgb = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bpl = ch * w
        qimg = QImage(frame_rgb.data, w, h, bpl, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_3d_view(self):
        self.set_status("Processing ...")
        if self.raw_image_data is None or self.num_frames <= 1:
            print("No multi-frame image loaded; cannot update 3D view.")
            return
        try:
            vol = np.clip(self.contrast * self.raw_image_data.astype(np.float32) + self.brightness, 0, 255).astype(np.uint8)
            if vol.ndim != 3:
                print("Volume data is not 3D; shape =", vol.shape)
                return
            print("Original volume shape (Z, Y, X):", vol.shape)
            importer = vtk.vtkImageImport()
            vol_contig = np.ascontiguousarray(vol)
            dims = vol_contig.shape
            importer.CopyImportVoidPointer(vol_contig.tobytes(), len(vol_contig.tobytes()))
            importer.SetDataScalarTypeToUnsignedChar()
            importer.SetNumberOfScalarComponents(1)
            importer.SetDataExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
            importer.SetWholeExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
            importer.SetDataSpacing(1.0, 1.0, 1.0)
            importer.SetDataOrigin(0.0, 0.0, 0.0)
            importer.Update()
            permute = vtk.vtkImagePermute()
            permute.SetInputConnection(importer.GetOutputPort())
            permute.SetFilteredAxes(2, 1, 0)
            permute.Update()
            flip = vtk.vtkImageFlip()
            flip.SetFilteredAxis(1)
            flip.SetInputConnection(permute.GetOutputPort())
            flip.Update()
            input_connection = flip.GetOutputPort()

            if self.use_gpu and self.is_gpu_supported():
                mapper = vtk.vtkGPUVolumeRayCastMapper()
                mapper.SetInputConnection(input_connection)
            else:
                mapper = vtk.vtkSmartVolumeMapper()
                mapper.SetInputConnection(input_connection)

            volumeProperty = vtk.vtkVolumeProperty()
            volumeProperty.ShadeOn()
            volumeProperty.SetAmbient(0.3)
            volumeProperty.SetDiffuse(0.7)
            volumeProperty.SetSpecular(0.3)
            volumeProperty.SetInterpolationTypeToLinear()
            opacityTransferFunction = vtk.vtkPiecewiseFunction()
            opacityTransferFunction.AddPoint(0, 0.0)
            opacityTransferFunction.AddPoint(32, 0.05)
            opacityTransferFunction.AddPoint(96, 0.15)
            opacityTransferFunction.AddPoint(160, 0.4)
            opacityTransferFunction.AddPoint(255, 0.8)
            volumeProperty.SetScalarOpacity(opacityTransferFunction)
            colorTransferFunction = vtk.vtkColorTransferFunction()
            colorTransferFunction.AddRGBPoint(0, 0.0, 0.0, 0.0)
            colorTransferFunction.AddRGBPoint(255, 1.0, 1.0, 1.0)
            volumeProperty.SetColor(colorTransferFunction)
            volumeActor = vtk.vtkVolume()
            volumeActor.SetMapper(mapper)
            volumeActor.SetProperty(volumeProperty)
            self.vtk_renderer.RemoveAllViewProps()
            self.vtk_renderer.AddVolume(volumeActor)
            self.vtk_renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            print("3D view updated successfully.")
            self.set_status("Ready", 2000)
        except Exception as e:
            print("Error in update_3d_view:", e)
            self.set_status("Error in 3D view update", 3000)

    def change_frame(self, value):
        self.current_frame_index = value
        self.z_frame_label.setText(f"Frame: {value+1}/{self.num_frames}")
        if self.current_view_mode == "2D":
            self.adjust_image()

    def get_current_frame(self, image_data):
        if image_data is None:
            return None
        if image_data.ndim == 2:
            return image_data
        elif image_data.ndim == 3:
            return image_data if image_data.shape[2] in (3,4) else image_data[self.current_frame_index]
        elif image_data.ndim == 4:
            return image_data[self.current_frame_index]
        else:
            return image_data

    def handle_mouse_move(self, event):
        if self.adjusted_image_data is None:
            return
        pos = event.pos()
        lw = self.image_label.width()
        lh = self.image_label.height()
        pix = self.image_label.pixmap()
        if pix is None:
            return
        if self.adjusted_image_data.ndim == 2:
            orig_h, orig_w = self.adjusted_image_data.shape
        else:
            orig_h, orig_w = self.adjusted_image_data.shape[:2]
        scale = min(lw / orig_w, lh / orig_h)
        disp_w = orig_w * scale
        disp_h = orig_h * scale
        offset_x = (lw - disp_w) / 2
        offset_y = (lh - disp_h) / 2
        x = int((pos.x() - offset_x) / scale)
        y = int((pos.y() - offset_y) / scale)
        if x < 0 or y < 0 or x >= orig_w or y >= orig_h:
            self.pixel_value_label.setText("Pixel: N/A")
            return
        value = self.adjusted_image_data[y, x] if self.adjusted_image_data.ndim == 2 else self.adjusted_image_data[y, x]
        self.pixel_value_label.setText(f"Pixel ({x}, {y}): {value}")

    def reset_adjustments(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.add_log("Reset brightness and contrast adjustments.")
        self.adjust_image()

    def auto_adjust_image(self):
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        if base is None:
            return
        frame = self.get_current_frame(base).astype(np.float32)
        min_val = np.min(frame)
        max_val = np.max(frame)
        if max_val - min_val == 0:
            contrast = 1.0
            brightness = 0
        else:
            contrast = 255.0 / (max_val - min_val)
            brightness = -min_val * contrast
        new_brightness = int(np.clip(brightness, -100, 100))
        new_contrast = int(np.clip(contrast * 100, 10, 300))
        self.brightness_slider.setValue(new_brightness)
        self.contrast_slider.setValue(new_contrast)
        self.add_log(f"Auto adjusted brightness and contrast: brightness={new_brightness}, contrast={new_contrast/100.0}")
        self.adjust_image()

    def toggle_view(self):
        if self.current_view_mode == "2D":
            if self.num_frames > 1 and self.raw_image_data is not None:
                self.current_view_mode = "3D"
                self.display_stack.setCurrentIndex(1)
                self.toggle_button.setText("Switch to 2D View")
                self.update_3d_view()
            else:
                QMessageBox.information(self, "3D View Not Available", "3D view is only available for multi-frame (z-stack) images.")
        else:
            self.current_view_mode = "2D"
            self.display_stack.setCurrentIndex(0)
            self.toggle_button.setText("Switch to 3D View")

    def filter_surfaces(self):
        if self.current_polydata is None:
            QMessageBox.warning(self, "No Surfaces", "No 3D surfaces available to filter.")
            return
        dialog = FilterSurfacesDialog(self.current_polydata, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.current_polydata = dialog.filtered_polydata
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.current_polydata)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.0, 0.0)
            self.vtk_renderer.RemoveAllViewProps()
            self.vtk_renderer.AddActor(actor)
            self.vtk_renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

    def delete_surfaces(self):
        if self.current_polydata is None:
            QMessageBox.warning(self, "No Surface", "No 3D surface available to filter.")
            return
        dialog = FilterSurfacesDialog(self.current_polydata, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.current_polydata = dialog.filtered_polydata
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.current_polydata)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.5, 0.5)
            self.vtk_renderer.RemoveAllViewProps()
            self.vtk_renderer.AddActor(actor)
            self.vtk_renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.add_log("Deleted surfaces outside the selected range.")

    def update_ml_slice_menu(self):
        self.ml_slice_menu.clear()
        for frame in sorted(self.ml_training_data.keys()):
            action = QAction(f"Frame {frame}", self)
            action.triggered.connect(lambda checked, f=frame: self.ml_classify_stack_slice(f))
            self.ml_slice_menu.addAction(action)

    # --- ML Classification Methods ---
    def ml_pixel_classifier(self):
        base = self.adjusted_image_data
        if base is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        frame = self.get_current_frame(base)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame.copy()
        dialog = MLPixelClassifierDialog(frame_gray, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            if self.ml_X_train is None:
                self.ml_X_train = dialog.training_data
                self.ml_y_train = dialog.training_labels
            else:
                self.ml_X_train = np.concatenate((self.ml_X_train, dialog.training_data), axis=0)
                self.ml_y_train = np.concatenate((self.ml_y_train, dialog.training_labels), axis=0)
            self.ml_training_data[self.current_frame_index] = (dialog.training_data, dialog.training_labels)
            new_clf = RandomForestClassifier(n_estimators=50)
            new_clf.fit(self.ml_X_train, self.ml_y_train)
            self.ml_classifier = new_clf
            mask = (dialog.result_segmentation[:, :, 0] == 0).astype(np.uint8) * 255
            self.roi_results[self.current_frame_index] = {"mask": mask, "contour": None}
            self.update_image_display()
            QMessageBox.information(self, "Segmentation", "ML segmentation applied to current slice.\n(Combined training data is now used.)")
            self.update_ml_slice_menu()

    def ml_classify_slice(self, frame_index):
        if frame_index not in self.ml_training_data:
            QMessageBox.warning(self, "No Training Data", f"No training data available for frame {frame_index}.")
            return
        X_train, y_train = self.ml_training_data[frame_index]
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X_train, y_train)
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        if base is None:
            return
        num_slices = 1 if base.ndim == 2 or (base.ndim == 3 and base.shape[2] in (3,4)) else base.shape[0]
        seg_stack = []
        for i in range(num_slices):
            slice_i = self.get_current_frame(base) if num_slices == 1 else base[i]
            gray = cv2.cvtColor(slice_i, cv2.COLOR_RGB2GRAY) if (slice_i.ndim == 3 and slice_i.shape[2] not in (3,4)) else slice_i.copy()
            h, w = gray.shape
            X_all = gray.reshape(-1, 1).astype(np.float32)
            y_pred = clf.predict(X_all).reshape(h, w)
            seg_color = np.zeros((h, w, 3), dtype=np.uint8)
            seg_color[y_pred == 1] = [0, 255, 0]
            seg_color[y_pred == 0] = [255, 0, 255]
            seg_stack.append(seg_color)
            self.roi_results[i] = {"mask": (y_pred==1).astype(np.uint8)*255, "contour": None}
        self.ml_segmentation_stack = seg_stack
        self.update_image_display()
        QMessageBox.information(self, "Slice Classification", f"ML classification using training from frame {frame_index} applied to entire stack.")

    def ml_classify_stack_combined(self):
        self.set_status("Processing ML classification for entire stack...")
        if not hasattr(self, "ml_classifier"):
            QMessageBox.warning(self, "No Classifier", "Please run the ML Pixel Classifier on the current slice first.")
            return
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        if base is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            self.set_status("Ready", 2000)
            return
        num_slices = 1 if base.ndim == 2 or (base.ndim == 3 and base.shape[2] in (3,4)) else base.shape[0]
        seg_stack = []
        for i in range(num_slices):
            slice_i = self.get_current_frame(base) if num_slices == 1 else base[i]
            gray = cv2.cvtColor(slice_i, cv2.COLOR_RGB2GRAY) if (slice_i.ndim == 3 and slice_i.shape[2] not in (3,4)) else slice_i.copy()
            h, w = gray.shape
            X_all = gray.reshape(-1, 1).astype(np.float32)
            y_pred = self.ml_classifier.predict(X_all).reshape(h, w)
            seg_color = np.zeros((h, w, 3), dtype=np.uint8)
            seg_color[y_pred == 1] = [0, 255, 0]
            seg_color[y_pred == 0] = [255, 0, 255]
            seg_stack.append(seg_color)
            self.roi_results[i] = {"mask": (y_pred==1).astype(np.uint8)*255, "contour": None}
        self.ml_segmentation_stack = seg_stack
        self.update_image_display()
        QMessageBox.information(self, "Stack Classification", "ML classification (combined training) applied to entire stack.")
        self.set_status("Ready", 2000)

    def ml_classify_stack_slice(self, frame_index):
        if frame_index not in self.ml_training_data:
            QMessageBox.warning(self, "No Training Data", f"No training data available for frame {frame_index}.")
            return
        X_train, y_train = self.ml_training_data[frame_index]
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X_train, y_train)
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        if base is None:
            return
        num_slices = 1 if base.ndim == 2 or (base.ndim == 3 and base.shape[2] in (3,4)) else base.shape[0]
        seg_stack = []
        for i in range(num_slices):
            slice_i = self.get_current_frame(base) if num_slices == 1 else base[i]
            gray = cv2.cvtColor(slice_i, cv2.COLOR_RGB2GRAY) if (slice_i.ndim == 3 and slice_i.shape[2] not in (3,4)) else slice_i.copy()
            h, w = gray.shape
            X_all = gray.reshape(-1, 1).astype(np.float32)
            y_pred = clf.predict(X_all).reshape(h, w)
            seg_color = np.zeros((h, w, 3), dtype=np.uint8)
            seg_color[y_pred == 1] = [0, 255, 0]
            seg_color[y_pred == 0] = [255, 0, 255]
            seg_stack.append(seg_color)
            self.roi_results[i] = {"mask": (y_pred==1).astype(np.uint8)*255, "contour": None}
        self.ml_segmentation_stack = seg_stack
        self.update_image_display()
        QMessageBox.information(self, "Slice Classification", f"ML classification using training from frame {frame_index} applied to entire stack.")

    def update_ml_slice_menu(self):
        self.ml_slice_menu.clear()
        for frame in sorted(self.ml_training_data.keys()):
            action = QAction(f"Frame {frame}", self)
            action.triggered.connect(lambda checked, f=frame: self.ml_classify_stack_slice(f))
            self.ml_slice_menu.addAction(action)

    # --- Standard methods for rendering, adjustment, etc. ---
    def render_roi_surface(self):
        self.set_status("Processing ...")
        # If a 3D surface was loaded from file, render it:
        if self.current_polydata is not None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.current_polydata)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.5, 0.5)
            self.vtk_renderer.RemoveAllViewProps()
            self.vtk_renderer.AddActor(actor)
            self.vtk_renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.set_status("Ready", 2000)
            return
        if not self.roi_results:
            QMessageBox.warning(self, "No ROI", "No ROI outlines available for rendering.")
            return
        if self.raw_image_data is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        if self.raw_image_data.ndim == 2:
            height, width = self.raw_image_data.shape; num_frames = 1
        elif self.raw_image_data.ndim == 3:
            if self.raw_image_data.shape[2] in (3,4):
                height, width = self.raw_image_data.shape[:2]; num_frames = 1
            else:
                num_frames, height, width = self.raw_image_data.shape
        elif self.raw_image_data.ndim == 4:
            num_frames, height, width = self.raw_image_data.shape[0], self.raw_image_data.shape[1], self.raw_image_data.shape[2]
        else:
            return
        vol = np.zeros((num_frames, height, width), dtype=np.uint8)
        for i in range(num_frames):
            if i in self.roi_results:
                vol[i] = self.roi_results[i]["mask"]
        vol = np.ascontiguousarray(vol)
        dims = vol.shape
        importer = vtk.vtkImageImport()
        importer.CopyImportVoidPointer(vol.tobytes(), len(vol.tobytes()))
        importer.SetDataScalarTypeToUnsignedChar()
        importer.SetNumberOfScalarComponents(1)
        importer.SetDataExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
        importer.SetWholeExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
        importer.Update()
        marchingCubes = vtk.vtkMarchingCubes()
        marchingCubes.SetInputConnection(importer.GetOutputPort())
        marchingCubes.SetValue(0, 127)
        marchingCubes.Update()
        num_points = marchingCubes.GetOutput().GetNumberOfPoints()
        if num_points == 0:
            QMessageBox.warning(self, "Thresholding", "No surface was extracted. Try a different threshold value.")
            return
        polyMapper = vtk.vtkPolyDataMapper()
        polyMapper.SetInputConnection(marchingCubes.GetOutputPort())
        polyMapper.ScalarVisibilityOff()
        surfaceActor = vtk.vtkActor()
        surfaceActor.SetMapper(polyMapper)
        surfaceActor.GetProperty().SetColor(1.0, 0.5, 0.5)
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(surfaceActor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.set_status("Ready", 2000)

    def render_final_structure(self):
        self.set_status("Processing ...")
        # If a final structure was loaded from file, render it:
        if self.final_polydata is not None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.final_polydata)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 1.0, 0.0)
            self.vtk_renderer.RemoveAllViewProps()
            self.vtk_renderer.AddActor(actor)
            self.vtk_renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.set_status("Ready", 2000)
            return
        if not self.roi_results:
            QMessageBox.warning(self, "No ROI", "No ROI outlines available for final structure.")
            return
        if self.raw_image_data is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        if self.raw_image_data.ndim == 2:
            height, width = self.raw_image_data.shape; num_frames = 1
        elif self.raw_image_data.ndim == 3:
            if self.raw_image_data.shape[2] in (3,4):
                height, width = self.raw_image_data.shape[:2]; num_frames = 1
            else:
                num_frames, height, width = self.raw_image_data.shape
        elif self.raw_image_data.ndim == 4:
            num_frames, height, width = self.raw_image_data.shape[0], self.raw_image_data.shape[1], self.raw_image_data.shape[2]
        else:
            return
        vol = np.zeros((num_frames, height, width), dtype=np.uint8)
        for i in range(num_frames):
            if i in self.roi_results:
                vol[i] = self.roi_results[i]["mask"]
        vol_bool = vol > 0
        closed = binary_closing(vol_bool, structure=np.ones((3,3,3)))
        filled = binary_fill_holes(closed)
        vol_final = (filled * 255).astype(np.uint8)
        vol_final = np.ascontiguousarray(vol_final)
        dims = vol_final.shape
        importer = vtk.vtkImageImport()
        importer.CopyImportVoidPointer(vol_final.tobytes(), len(vol_final.tobytes()))
        importer.SetDataScalarTypeToUnsignedChar()
        importer.SetNumberOfScalarComponents(1)
        importer.SetDataExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
        importer.SetWholeExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
        importer.Update()
        marchingCubes = vtk.vtkMarchingCubes()
        marchingCubes.SetInputConnection(importer.GetOutputPort())
        marchingCubes.SetValue(0, 127)
        marchingCubes.Update()
        num_points = marchingCubes.GetOutput().GetNumberOfPoints()
        if num_points == 0:
            QMessageBox.warning(self, "Final Structure", "No surface was extracted. Please check your ROI outlines.")
            return
        polyMapper = vtk.vtkPolyDataMapper()
        polyMapper.SetInputConnection(marchingCubes.GetOutputPort())
        polyMapper.ScalarVisibilityOff()
        surfaceActor = vtk.vtkActor()
        surfaceActor.SetMapper(polyMapper)
        surfaceActor.GetProperty().SetColor(1.0, 1.0, 0.0)
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(surfaceActor)
        self.vtk_renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.set_status("Ready", 2000)

    # --- Image adjustment and display methods ---
    def adjust_image(self):
        if self.raw_image_data is None:
            return
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        frame = self.get_current_frame(base)
        frame_float = frame.astype(np.float32)
        self.brightness = self.brightness_slider.value()
        self.contrast = self.contrast_slider.value() / 100.0
        adjusted = np.clip(self.contrast * frame_float + self.brightness, 0, 255).astype(np.uint8)
        self.adjusted_image_data = adjusted
        self.update_image_display()
        self.add_log(f"Applied brightness/contrast: brightness={self.brightness}, contrast={self.contrast}")
        if self.current_view_mode == "3D" and self.num_frames > 1:
            self.update_3d_view()

    def update_image_display(self):
        if self.adjusted_image_data is None:
            return
        frame = self.get_current_frame(self.adjusted_image_data).copy()
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        if self.roi_results and self.current_frame_index in self.roi_results:
            contour = self.roi_results[self.current_frame_index]["contour"]
            cv2.drawContours(frame_color, [contour], -1, (0,255,255), 2)
        frame_rgb = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bpl = ch * w
        qimg = QImage(frame_rgb.data, w, h, bpl, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_3d_view(self):
        self.set_status("Processing ...")
        if self.raw_image_data is None or self.num_frames <= 1:
            print("No multi-frame image loaded; cannot update 3D view.")
            return
        try:
            vol = np.clip(self.contrast * self.raw_image_data.astype(np.float32) + self.brightness, 0, 255).astype(np.uint8)
            if vol.ndim != 3:
                print("Volume data is not 3D; shape =", vol.shape)
                return
            print("Original volume shape (Z, Y, X):", vol.shape)
            importer = vtk.vtkImageImport()
            vol_contig = np.ascontiguousarray(vol)
            dims = vol_contig.shape
            importer.CopyImportVoidPointer(vol_contig.tobytes(), len(vol_contig.tobytes()))
            importer.SetDataScalarTypeToUnsignedChar()
            importer.SetNumberOfScalarComponents(1)
            importer.SetDataExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
            importer.SetWholeExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
            importer.SetDataSpacing(1.0, 1.0, 1.0)
            importer.SetDataOrigin(0.0, 0.0, 0.0)
            importer.Update()
            permute = vtk.vtkImagePermute()
            permute.SetInputConnection(importer.GetOutputPort())
            permute.SetFilteredAxes(2, 1, 0)
            permute.Update()
            flip = vtk.vtkImageFlip()
            flip.SetFilteredAxis(1)
            flip.SetInputConnection(permute.GetOutputPort())
            flip.Update()
            input_connection = flip.GetOutputPort()
            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputConnection(input_connection)
            volumeProperty = vtk.vtkVolumeProperty()
            volumeProperty.ShadeOn()
            volumeProperty.SetAmbient(0.3)
            volumeProperty.SetDiffuse(0.7)
            volumeProperty.SetSpecular(0.3)
            volumeProperty.SetInterpolationTypeToLinear()
            opacityTransferFunction = vtk.vtkPiecewiseFunction()
            opacityTransferFunction.AddPoint(0, 0.0)
            opacityTransferFunction.AddPoint(32, 0.05)
            opacityTransferFunction.AddPoint(96, 0.15)
            opacityTransferFunction.AddPoint(160, 0.4)
            opacityTransferFunction.AddPoint(255, 0.8)
            volumeProperty.SetScalarOpacity(opacityTransferFunction)
            colorTransferFunction = vtk.vtkColorTransferFunction()
            colorTransferFunction.AddRGBPoint(0, 0.0, 0.0, 0.0)
            colorTransferFunction.AddRGBPoint(255, 1.0, 1.0, 1.0)
            volumeProperty.SetColor(colorTransferFunction)
            volumeActor = vtk.vtkVolume()
            volumeActor.SetMapper(mapper)
            volumeActor.SetProperty(volumeProperty)
            self.vtk_renderer.RemoveAllViewProps()
            self.vtk_renderer.AddVolume(volumeActor)
            self.vtk_renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            print("3D view updated successfully.")
            self.set_status("Ready", 2000)
        except Exception as e:
            print("Error in update_3d_view:", e)
            self.set_status("Error in 3D view update", 3000)

    def change_frame(self, value):
        self.current_frame_index = value
        self.z_frame_label.setText(f"Frame: {value+1}/{self.num_frames}")
        if self.current_view_mode == "2D":
            self.adjust_image()

    def get_current_frame(self, image_data):
        if image_data is None:
            return None
        if image_data.ndim == 2:
            return image_data
        elif image_data.ndim == 3:
            return image_data if image_data.shape[2] in (3,4) else image_data[self.current_frame_index]
        elif image_data.ndim == 4:
            return image_data[self.current_frame_index]
        else:
            return image_data

    def handle_mouse_move(self, event):
        if self.adjusted_image_data is None:
            return
        pos = event.pos()
        lw = self.image_label.width()
        lh = self.image_label.height()
        pix = self.image_label.pixmap()
        if pix is None:
            return
        if self.adjusted_image_data.ndim == 2:
            orig_h, orig_w = self.adjusted_image_data.shape
        else:
            orig_h, orig_w = self.adjusted_image_data.shape[:2]
        scale = min(lw / orig_w, lh / orig_h)
        disp_w = orig_w * scale
        disp_h = orig_h * scale
        offset_x = (lw - disp_w) / 2
        offset_y = (lh - disp_h) / 2
        x = int((pos.x() - offset_x) / scale)
        y = int((pos.y() - offset_y) / scale)
        if x < 0 or y < 0 or x >= orig_w or y >= orig_h:
            self.pixel_value_label.setText("Pixel: N/A")
            return
        value = self.adjusted_image_data[y, x] if self.adjusted_image_data.ndim == 2 else self.adjusted_image_data[y, x]
        self.pixel_value_label.setText(f"Pixel ({x}, {y}): {value}")

    def reset_adjustments(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.add_log("Reset brightness and contrast adjustments.")
        self.adjust_image()

    def auto_adjust_image(self):
        base = self.raw_image_data if self.show_raw else self.corrected_image_data
        if base is None:
            return
        frame = self.get_current_frame(base).astype(np.float32)
        min_val = np.min(frame)
        max_val = np.max(frame)
        if max_val - min_val == 0:
            contrast = 1.0
            brightness = 0
        else:
            contrast = 255.0 / (max_val - min_val)
            brightness = -min_val * contrast
        new_brightness = int(np.clip(brightness, -100, 100))
        new_contrast = int(np.clip(contrast * 100, 10, 300))
        self.brightness_slider.setValue(new_brightness)
        self.contrast_slider.setValue(new_contrast)
        self.add_log(f"Auto adjusted brightness and contrast: brightness={new_brightness}, contrast={new_contrast/100.0}")
        self.adjust_image()

# --- End of ImageApp class ---

if __name__ == "__main__":
    def main():
        app = QApplication(sys.argv)
        window = ImageApp()
        window.show()
        sys.exit(app.exec_())
    main()
