import cv2
import numpy as np

checkerboard_size = (4, 4) 
objpoints = [] 
imgpoints1 = []  
imgpoints2 = []  

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

def find_available_cameras():
    available_cameras = []
    for i in range(10):  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def calibration_complete():
    return len(objpoints) >= 10  

available_cameras = find_available_cameras()

if len(available_cameras) >= 2:
    cap1 = cv2.VideoCapture(available_cameras[0])  
    cap2 = cv2.VideoCapture(available_cameras[1])  

    gray1 = None
    gray2 = None

    calibration_complete_flag = False

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Failed to read frames from the cameras")
            break

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        ret1, corners1 = cv2.findChessboardCorners(gray1, checkerboard_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, checkerboard_size, None)

        if ret1 and ret2:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv2.drawChessboardCorners(frame1, checkerboard_size, corners1, ret1)
            cv2.drawChessboardCorners(frame2, checkerboard_size, corners2, ret2)

            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

            if calibration_complete():
                calibration_complete_flag = True

        cv2.imshow("Camera 1 Calibration", frame1)
        cv2.imshow("Camera 2 Calibration", frame2)

        if cv2.waitKey(1) & 0xFF == ord('q') or calibration_complete_flag:
            break
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    if calibration_complete_flag:
        print("Captured calibration!")
        ret, camera_matrix1, dist_coeff1, camera_matrix2, dist_coeff2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2, None, None, None, None,
            (gray1.shape[1], gray1.shape[0]), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        np.save("camera_matrix_1.npy", camera_matrix1)
        np.save("dist_coeff_1.npy", dist_coeff1)
        np.save("camera_matrix_2.npy", camera_matrix2)
        np.save("dist_coeff_2.npy", dist_coeff2)
        np.save("rotation_matrix.npy", R)
        np.save("translation_vector.npy", T)

        print("Calibration complete. Calibration parameters saved to files.")
    else:
        print("Calibration failed. Make sure the cameras are properly connected and capturing frames.")
else:
    print("Insufficient number of available cameras. Connect at least two cameras.")
