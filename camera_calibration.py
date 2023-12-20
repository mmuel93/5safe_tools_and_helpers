import cv2 as cv
import numpy as np
import glob
import yaml
import json
import matplotlib.pyplot as plt

def calculate_camera_calibration_matirx(folderpath, image_format, chessboard_format):
    """
        Calculates a camera calibration Matrix from checkered board Images
        Input: Path to dir with Calibration Images, Chessboard Corners Format (b x h)
        Output: Intrinsic Camera Parameters Matrix
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_format[0] * chessboard_format[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_format[0],0:chessboard_format[1]].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    if (image_format == "png"):
        images = glob.glob(folderpath + '/*.png')
    if (image_format == "jpg"):
        images = glob.glob(folderpath + '/*.JPG')
    #else:
    #    raise Exception("No Input Images in Specified Format found! Check Format or Inputfolder")
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        print(fname)
        ret, corners = cv.findChessboardCorners(gray, chessboard_format, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            #cv.drawChessboardCorners(img, chessboard_format, corners2, ret)
            #cv.imshow(fname, img)
            #cv.waitKey()
        else:
            raise Exception ("No Chessboard Corners found, please check Input Images and try again!")
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist, rvecs, tvecs

def undistort_webcam_stream(width, height, camera_matrix, distortion_coeffs):
    """
    Undistorts a Webcam Stream and displays it. Press Esc to stop
    Input: Stream Dimensions (width, height)
           Camera Calibration Matrix (3x3)
           Distortion Coefficients (5x1)
    Output: Undistorted Webcam Stream
    """
    DIM = (height, width)

    cap = cv.VideoCapture(0)
    if cap is None:
        raise Exception ("No Webcam Stream detected, please check Video Input!")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, DIM[1]) 
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, DIM[0]) 
    while True:
            flag, img = cap.read()
            try:
            
                h,  w = DIM[0], DIM[1]
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w,h), 1, (w,h))
                # undistort
                mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, distortion_coeffs, None, newcameramtx, (w,h), 5)
                dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
                # crop the image
                #x, y, w, h = roi
                #dst = dst[y:y+h, x:x+w]
                cv.imshow('calibresult.jpg', dst)
            except:
                cap.release()
                raise
            k = cv.waitKey(1)
            if k == 27:
                break
    cap.release()
    cv.destroyAllWindows()

def dump_camera_information_to_yaml(folderpath, filename, cam_dict):
    with open(folderpath + '/' + filename +'.yml', 'w') as outfile:
        yaml.dump(cam_dict, outfile, default_flow_style=False)

def dump_camera_information_to_json(folderpath, filename, cam_dict):
    with open(folderpath + "/" + filename, 'w') as fp:
        json.dump(cam_dict, fp)


if __name__ == '__main__':
   mat, dist, rot, trans = calculate_camera_calibration_matirx("C:/Users/mum21730/Projekte/Getting_Started/calibration_images/IP_camera_VUP/", "png", (8, 5))
   cam_dict = {"Mat": mat, "Dist": dist, "Rot": rot, "Trans": trans}

   img_to_undist = cv.imread("C:/Users/mum21730/Desktop/5_Safe/Bilder/petpa/camera_1/keyframe_camera1.jpg")

   img_undist = cv.undistort(img_to_undist, mat, dist)
   cv.imwrite("C:/Users/mum21730/Desktop/5_Safe/Bilder/petpa/camera_1/undist_keyframe_camera1.jpg", img_undist)
   img_to_undist = cv.cvtColor(img_undist, cv.COLOR_BGR2RGB)
   plt.imshow(img_undist)
   plt.show()
   
   #dump_camera_information_to_yaml("camera_matrices", "IP_Camera_VUP", cam_dict)
   #dump_camera_information_to_json("camera_matrices", "DJI_Mini_3_Marcel", cam_dict)
   print("Camera_Matrix:")
   print(mat)
   print("Distortion Coefficients:")
   print(dist)
   #print(rot)
   #print(trans)
   #undistort_webcam_stream(1280, 720, mat, dist)
   
   
