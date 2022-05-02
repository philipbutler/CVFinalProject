# Jiapei Li
# CS 5330
# Final Project
# Visualization

# import statements
import cv2 as cv
import glob
import numpy as np

# initial filter parameters
# filter_path = 'filters/witch.png'
# filter_path = 'filters/pikachu_filter.png'

CACHE_PTS = None

# filter class
class Filter:
    def __init__(self, filter, ori_filter_h, ori_filter_w, mask, mask_inv) -> None:
        self.filter = filter
        self.ori_filter_h = ori_filter_h
        self.ori_filter_w = ori_filter_w
        self.mask = mask
        self.mask_inv = mask_inv

    # getters
    def getFilterPackage(self):
        return self.filter, self.ori_filter_h, self.ori_filter_w, self.mask, self.mask_inv

# switch case to get specific height and width parameters for filters
def getFilterParameters(classifer):
    match classifer:
        # pikachu
        case 0:
            return 1.82, 0.4
        # witch
        case 1:
            return 2.0, 1.3
        case _:
            return 1.82, 0.4

# load filter and get masks
def loadFilters():
    filter_dir = glob.glob('filters/*.png')
    filters = []

    for filter_path in filter_dir:
        filter = cv.imread(filter_path)
        filter_unchanged = cv.imread(filter_path, cv.IMREAD_UNCHANGED)

        # filter mask
        # filter_gray = cv.cvtColor(filter, cv.COLOR_BGR2GRAY)
        ori_filter_h, ori_filter_w, filter_channels = filter.shape

        # cv.THRESH_BINARY_INV for png transparent background, cv.THRESH_BINARY for white background
        # ret, mask = cv.threshold(filter_gray, 10, 255, cv.THRESH_BINARY_INV)

        # Reference - https://stackoverflow.com/questions/48816703/opencv-turn-transparent-part-of-png-white
        ret, mask = cv.threshold(
            filter_unchanged[:, :, 3], 10, 255, cv.THRESH_BINARY_INV)
        mask_inv = cv.bitwise_not(mask)

        # work in progress
        # cv.imshow("trans", filter[trans_mask])
        # cv.imshow("gray", filter_gray)
        # cv.imshow("mask", mask)
        # cv.imshow("inv", mask_inv)

        filter_curr = Filter(filter, ori_filter_h,
                             ori_filter_w, mask, mask_inv)
        filters.append(filter_curr)

    return filters


# apply filter to face region
def applyFilter(frame, faces, filters, counter):
    # getters
    curr_filter = filters[counter % len(filters)]
    filter, ori_filter_h, ori_filter_w, mask, mask_inv = curr_filter.getFilterPackage()

    # loop through every face found
    for (x, y, w, h) in faces:
        # get coordinates of 4 corners, and w & h
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        # scale filter to fit face
        filter_width = int(1.5 * face_w)
        # same proportion
        filter_height = int(filter_width * (ori_filter_h / ori_filter_w))

        # switch case
        width_factor, height_factor = getFilterParameters(
            counter % len(filters))

        # get center of face, deducted by half of filter width
        filter_x1 = face_x2 - int(face_w / 2) - \
            int(filter_width / width_factor)
        filter_x2 = filter_x1 + filter_width
        # face's upper left -> filter's upper left, adjustable factor
        filter_y1 = face_y1 - int(face_h * height_factor)
        filter_y2 = filter_y1 + filter_height

        # check boundaries
        frame_height, frame_width, frame_channels = frame.shape
        if filter_x1 < 0:
            filter_x1 = 0
        if filter_x2 > frame_width:
            filter_x2 = frame_width
        if filter_y1 < 0:
            filter_y1 = 0
        if filter_y2 > frame_height:
            filter_y2 = frame_height

        # update filter region, if there is out of boundary condition
        filter_width = filter_x2 - filter_x1
        filter_height = filter_y2 - filter_y1

        # resize filter, and fit to region
        filter = cv.resize(filter, (filter_width, filter_height),
                           interpolation=cv.INTER_AREA)
        mask = cv.resize(mask, (filter_width, filter_height),
                         interpolation=cv.INTER_AREA)
        mask_inv = cv.resize(
            mask_inv, (filter_width, filter_height), interpolation=cv.INTER_AREA)

        # crop original region in the frame
        region = frame[filter_y1:filter_y2, filter_x1:filter_x2]

        # get region's background and foreground using masks
        # Reference - https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
        region_bg = cv.bitwise_and(region, region, mask=mask)
        region_fg = cv.bitwise_and(filter, filter, mask=mask_inv)
        ret = cv.add(region_bg, region_fg)

        # project to frame
        frame[filter_y1:filter_y2, filter_x1:filter_x2] = ret

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
