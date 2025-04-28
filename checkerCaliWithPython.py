import cv2
import numpy as np
import glob
import os

def calibrate_cameras(image_dir, board_size, square_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3d points in real world space
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    images_left = glob.glob(os.path.join(image_dir, 'left_*.png'))
    images_right = glob.glob(os.path.join(image_dir, 'right_*.png'))

    for img_left, img_right in zip(images_left, images_right):
        imgL = cv2.imread(img_left)
        imgR = cv2.imread(img_right)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(grayL, (board_size[0], board_size[1]), None)
        retR, cornersR = cv2.findChessboardCorners(grayR, (board_size[0], board_size[1]), None)

        if retL and retR:
            objpoints.append(objp)

            corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners2L)

            corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgpoints_right.append(corners2R)

    # Calibrate each camera individually
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

    # Stereo calibration
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtxL, distL, mtxR, distR, grayL.shape[::-1], criteria, flags)

    # Rectification
    rectify_scale = 0  # 0: crop, 1: keep all pixels
    RL, RR, PL, PR, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T, alpha=rectify_scale)

    # Save calibration parameters
    np.savez('stereo_calibration_params.npz', 
             mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, RL=RL, RR=RR, PL=PL, PR=PR, Q=Q)

    print("Stereo calibration complete.")
    print(f"Left camera matrix:\n{mtxL}")
    print(f"Right camera matrix:\n{mtxR}")
    print(f"Rotation matrix:\n{R}")
    print(f"Translation vector:\n{T}")

# Calibrate with captured images
calibrate_cameras(r"C:\Users\Malavika\checkerboard_images", (8, 11), 20)

