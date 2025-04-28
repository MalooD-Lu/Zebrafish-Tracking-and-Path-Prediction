import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
import pandas as pd

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

# Open the stereo video files
cap1 = cv2.VideoCapture(r"C:\Users\Malavika\yolov7\2x2_1hr\2x2_1hr\output_left.mp4")
cap2 = cv2.VideoCapture(r"C:\Users\Malavika\yolov7\2x2_1hr\2x2_1hr\output_right.mp4")

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open video files.")
    exit()

# loading npz file
file_path = r"C:\Users\Malavika\yolov7\stereo_calibration_params_ayan.npz"
data = np.load(file_path)

# Extracting param
K1 = data['mtxL']
K2 = data['mtxR']
D1 = data['distL']
D2 = data['distR']
R = data['R']
T = data['T']

# Printing the param
print("K1:", K1)
print("K2:", K2)
print("D1:", D1)
print("D2:", D2)
print("R:", R)
print("T:", T)

# List to store fish coordinates
fish_3D_coords_list = []

while True:
    # Read frames from each video
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Perform detection on the frames
    results1 = run_inference(model, frame1)
    results2 = run_inference(model, frame2)

    # Rectify the images
    h, w = frame1.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    rectified1 = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR)
    rectified2 = cv2.remap(frame2, map2x, map2y, cv2.INTER_LINEAR)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(rectified1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rectified2, cv2.COLOR_BGR2GRAY)

    min_disparity = 159  # Based on the above calculation
    num_disparities = 476  # Should be multiple of 16

    if num_disparities % 16 != 0:
        num_disparities += 16 - (num_disparities % 16)
    # Compute disparity map
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,  # Multiple of 16
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity_map = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

    # Reproject points to 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

    # Extract 3D coordinates for detected fish
    fish_3D_coords = []

    for box in results1[0]:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cx = int((x1 + x2) / 2)  # Calculate the center of the bounding box
        cy = int((y1 + y2) / 2)
        fish_3D_coords.append(points_3D[cy, cx])  # Get the 3D coordinates at the center of the bounding box

    # Append the 3D coordinates to the list
    fish_3D_coords_list.extend(fish_3D_coords)

    print(fish_3D_coords)

    # Display the results
    for box in results1[0]:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Ensure the coordinates are integers
        label = f'{int(cls)} {conf:.2f}'
        cv2.rectangle(frame1, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('YOLO Detection', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

# Save 3D coordinates to an Excel file
df = pd.DataFrame(fish_3D_coords_list, columns=['X', 'Y', 'Z'])
df.to_excel(r"C:\Users\Malavika\yolov7\fish_3D_coords.xlsx", index=False)

