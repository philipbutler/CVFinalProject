# Jiapei Li
# CS 5330
# Final Project
# Visualization

# import statements
from lib2to3 import pygram
import cv2 as cv
import glob
import numpy as np
from PIL import Image
from pathlib import Path

# global variables
gif_path = 'filters/*.gif'
CACHE_PTS = None

# helper method to get new destination points or CACHED points
def getDstPts(corners, ids):
    global CACHE_PTS

    if ids is not None and len(ids) == 4:
        # upper left, corner 0
        index = np.squeeze(np.where(ids == 12))
        pt1 = np.squeeze(corners[index[0]])[0]

        # upper right, corner 1
        index = np.squeeze(np.where(ids == 22))
        pt2 = np.squeeze(corners[index[0]])[1]

        horizontal_distance = np.linalg.norm(pt2 - pt1)

        # extend the boundary from center of the marker to the edges
        scaling_factor = 0.02
        # upper left
        pts_dst = [[pt1[0] - round(horizontal_distance * scaling_factor),
                   pt1[1] - round(horizontal_distance * scaling_factor)]]
        # upper right
        pts_dst = pts_dst + [[pt2[0] + round(horizontal_distance * scaling_factor),
                             pt2[1] - round(horizontal_distance * scaling_factor)]]

        # bottom right, corner 2
        index = np.squeeze(np.where(ids == 32))
        pt3 = np.squeeze(corners[index[0]])[2]
        pts_dst = pts_dst + [[pt3[0] + round(horizontal_distance * scaling_factor),
                             pt3[1] + round(horizontal_distance * scaling_factor)]]

        # bottom left, corner 3
        index = np.squeeze(np.where(ids == 42))
        pt4 = np.squeeze(corners[index[0]])[3]
        pts_dst = pts_dst + [[pt4[0] - round(horizontal_distance * scaling_factor),
                             pt4[1] + round(horizontal_distance * scaling_factor)]]

        CACHE_PTS = pts_dst

        return pts_dst

    elif CACHE_PTS is not None:
        return CACHE_PTS
    else:
        return None


# detect Aruco Markers
def detectAndShowMarkers(frame, dictionary, image_src):
    # initiate the detector parameters
    parameters = cv.aruco.DetectorParameters_create()

    # detect the markers in the image
    corners, ids, rejectedCandidates = cv.aruco.detectMarkers(
        frame, dictionary, parameters=parameters)

    # if markerIds is not None:
    #     cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    #     rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
    #         markerCorners, 0.05, cameraMatrix, coeffs)

    #     for i in range(len(rvecs)):
    #         cv.drawFrameAxes(frame, cameraMatrix, coeffs,
    #                          rvecs[i], tvecs[i], 0.1)

    pts_dst = getDstPts(corners, ids)

    # directly return frame if it is None
    if pts_dst is None:
        return frame

    # destination points have been found
    else:
        # 0 as h, 1 as w -> 4 corners, upper left, upper right, bottom right, bottom left
        pts_src = [[0, 0], [image_src.shape[1], 0], [
            image_src.shape[1], image_src.shape[0]], [0, image_src.shape[0]]]

        pts_src_homo = np.asarray(pts_src)
        pts_dst_homo = np.asarray(pts_dst)

        # homography
        homo, status = cv.findHomography(pts_src_homo, pts_dst_homo)

        # calculate warpped image's shape
        warpped_image = cv.warpPerspective(
            image_src, homo, (frame.shape[1], frame.shape[0]))

        # mask's height and width
        mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
        # Mask as the region to copy from the mapped image into the original frame
        cv.fillConvexPoly(mask, np.int32(
            [pts_dst_homo]), (255, 255, 255), cv.LINE_AA)

        # Erode the mask to not copy the boundary effects from the mapping process
        element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        mask = cv.erode(mask, element, iterations=2)

        # convert into 3 channels
        warpped_image = warpped_image.astype(float)
        mask_three = np.zeros_like(warpped_image)
        for i in range(0, 3):
            mask_three[:, :, i] = mask / 255

        # warpped image -> mask
        warpped_image_after_mask = cv.multiply(warpped_image, mask_three)
        # frame -> mask's inversion
        frame_after_mask = cv.multiply(frame.astype(float), 1 - mask_three)
        frame = cv.add(warpped_image_after_mask,
                       frame_after_mask)
        return frame


# load camera calibration info
# Reference - https://stackoverflow.com/questions/35578405/python-how-to-read-line-by-line-in-a-numpy-array
def loadCameraCalibrationInfo(filename, cameraMatrix, coeffs):
    print("\nloading co-efficients info...")
    with open(filename) as f:
        r = 0
        for line in f:
            if r <= 2:
                nums = line.split(' ')
                # for c in range(0, 3):
                #     cameraMatrix[r, c] = nums[c]
                cameraMatrix[r:] = np.fromstring(
                    line, dtype=np.double, sep=' ')
            else:
                coeffs = np.fromstring(line, dtype=np.double, sep=' ')
            r = r + 1

    if len(coeffs) > 0:
        print("Successfully loaded camera calibration info.")
        print(cameraMatrix)
        print(coeffs)
    else:
        print("Failed loading camera calibration info.")
        exit()

# load GIF to image array
# Reference - Convert image from PIL to openCV format
# https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
def loadGifsToMap(filterMap):
    gif_dir = glob.glob(gif_path)

    for path in gif_dir:
        gif_name = Path(path).name.split('.')[0].split('_')[0]
        gif_frames = []

        # image_srcs = []
        im = Image.open(path)
        try:
            for index in range(im.n_frames):
                im.seek(index)
                # cannot be directly converted to BGR
                tmp = im.convert("RGB")
                a = np.array(tmp)
                a = cv.cvtColor(a, cv.COLOR_RGB2BGR)
                gif_frames.append(a)

        except EOFError:
            pass

        # assign
        filterMap[gif_name].gif = gif_frames
