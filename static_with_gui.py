import sys
import os
# Adjust the path to include YOLOv7 repository
yolov7_path = r"C:\Users\Malavika\yolov7"  # Change this to the path where you cloned YOLOv7
sys.path.append(yolov7_path)

import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSplitter
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# Adjust the path to include YOLOv7 repository
yolov7_path = r"C:\Users\Malavika\yolov7"  # Change this to the path where you cloned YOLOv7
sys.path.append(yolov7_path)

# Load the YOLOv7 model
model = attempt_load(r"C:\Users\Malavika\yolov7\best.pt", map_location='cpu')  # Load the model

# Function to perform inference
def run_inference(model, img):
    img_size = 640  # Define input size for YOLOv7
    stride = int(model.stride.max())  # Get stride size
    img = letterbox(img, img_size, stride=stride)[0]  # Resize image to fit model input size

    img = img.transpose(2, 0, 1)  # Convert HWC to CHW
    img = np.ascontiguousarray(img)  # Make image contiguous in memory

    img = torch.from_numpy(img).float()  # Convert numpy array to torch tensor
    img /= 255.0  # Normalize to [0, 1]
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # Disable gradient calculation
        pred = model(img, augment=False)[0]  # Run inference

    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)  # Apply NMS

    return pred

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('YOLOv7 and 3D Plot')
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Splitter to divide the video and plot sections
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # Video display label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        splitter.addWidget(self.video_label)

        # Matplotlib FigureCanvas for 3D plot
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        splitter.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-300, 700])
        self.ax.set_ylim([-150, 220])
        self.ax.set_zlim([1900, 2000])
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')

        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

        # Open video files
        self.cap1 = cv2.VideoCapture(r"C:\Users\Malavika\yolov7\2x2_1hr\2x2_1hr\output_left.mp4")
        self.cap2 = cv2.VideoCapture(r"C:\Users\Malavika\yolov7\2x2_1hr\2x2_1hr\output_right.mp4")

        if not self.cap1.isOpened() or not self.cap2.isOpened():
            print("Error: Could not open video files.")
            exit()

        # Load stereo calibration parameters
        file_path = r"C:\Users\Malavika\yolov7\stereo_calibration_params_ayan.npz"
        data = np.load(file_path)

        self.K1 = data['mtxL']
        self.K2 = data['mtxR']
        self.D1 = data['distL']
        self.D2 = data['distR']
        self.R = data['R']
        self.T = data['T']

        self.fish_3D_coords_list = []
        self.frame_number = 0
        self.frame_skip = 2

    def update_frame(self):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if not ret1 or not ret2:
            self.timer.stop()
            self.cap1.release()
            self.cap2.release()
            return

        self.frame_number += 1

        if self.frame_number % self.frame_skip != 0:
            return

        results1 = run_inference(model, frame1)
        results2 = run_inference(model, frame2)

        h, w = frame1.shape[:2]
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(self.K1, self.D1, self.K2, self.D2, (w, h), self.R, self.T)
        map1x, map1y = cv2.initUndistortRectifyMap(self.K1, self.D1, R1, P1, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(self.K2, self.D2, R2, P2, (w, h), cv2.CV_32FC1)

        rectified1 = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR)
        rectified2 = cv2.remap(frame2, map2x, map2y, cv2.INTER_LINEAR)

        gray1 = cv2.cvtColor(rectified1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(rectified2, cv2.COLOR_BGR2GRAY)

        min_disparity = 159
        num_disparities = 476
        if num_disparities % 16 != 0:
            num_disparities += 16 - (num_disparities % 16)

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        disparity_map = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
        points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

        fish_3D_coords = []
        for box in results1[0]:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if 0 <= cy < points_3D.shape[0] and 0 <= cx < points_3D.shape[1]:
                fish_3D_coords.append([self.frame_number, *points_3D[cy, cx]])

        self.fish_3D_coords_list.extend(fish_3D_coords)
        seen = set()
        unique_fish_3D_coords_list = []
        for coord in self.fish_3D_coords_list:
            if tuple(coord[1:]) not in seen:
                unique_fish_3D_coords_list.append(coord)
                seen.add(tuple(coord[1:]))

        for box in results1[0]:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f'{int(cls)} {conf:.2f}'
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        qimg = QImage(frame1.data, frame1.shape[1], frame1.shape[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)

        if unique_fish_3D_coords_list:
            fish_3D_coords_array = np.array(unique_fish_3D_coords_list)
            self.ax.clear()
            self.ax.set_xlim([-300, 700])
            self.ax.set_ylim([-150, 220])
            self.ax.set_zlim([1900, 2000])
            self.ax.set_xlabel('X Label')
            self.ax.set_ylabel('Y Label')
            self.ax.set_zlabel('Z Label')
            self.ax.scatter(fish_3D_coords_array[:, 1], fish_3D_coords_array[:, 2], fish_3D_coords_array[:, 3], c='r', marker='o')
            self.canvas.draw()

    def closeEvent(self, event):
        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

