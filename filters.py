# Jiapei Li
# CS 5330
# Final Project
# Visualization

# import statements
import cv2 as cv
import glob
import numpy as np
from pathlib import Path

# initial filter parameters
png_path = 'filters/*.png'


# filter class
class Filter:
    def __init__(self, label, filter, ori_filter_h, ori_filter_w, mask, mask_inv) -> None:
        self.label = label
        self.filter = filter
        self.ori_filter_h = ori_filter_h
        self.ori_filter_w = ori_filter_w
        self.mask = mask
        self.mask_inv = mask_inv
        self.gif = None

    # getters
    def getFilterPackage(self):
        return self.filter, self.ori_filter_h, self.ori_filter_w, self.mask, self.mask_inv

# switch case to get specific height and width parameters for filters
def getFilterParameters(label):
    # pikachu
    if label == 'pikachu':
        return 1.82, 0.4
    # witch
    elif label == 'witch':
        return 2.0, 1.3
    # charmander
    elif label == 'charmander':
        return 2.0, 0.4
    # kirby
    elif label == 'kirby':
        return 2.24, 0.4
    # jigglypuff
    elif label == 'jigglypuff':
        return 2.0, 1.2
    else:
        return 1.82, 0.4

# person name to filter label
def loadPersonNameToLabel():
    nameToLabel = {}
    nameToLabel['changling'] = 'jigglypuff'
    nameToLabel['phil'] = 'kirby'
    nameToLabel['erica'] = 'charmander'
    nameToLabel['jp'] = 'pikachu'
    nameToLabel['unknown'] = 'witch'

    return nameToLabel

# load filter and get masks
def loadFilters():
    filter_dir = glob.glob(png_path)
    filterMap = {}

    for filter_path in filter_dir:
        filter_name = Path(filter_path).name.split('.')[0]
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

        filter_curr = Filter(filter_name, filter,
                             ori_filter_h, ori_filter_w, mask, mask_inv)
        filterMap[filter_name] = filter_curr

    return filterMap


# apply filter to face region
def applyFilter(frame, faces, curr_filter):
    # getters
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
        width_factor, height_factor = getFilterParameters(curr_filter.label)

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
