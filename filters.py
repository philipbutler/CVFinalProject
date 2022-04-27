# import statements
import cv2 as cv

# initial filter parameters
# filter_path = 'filters/witch.png'
filter_path = 'filters/pikachu_filter-01.png'


# load filter and get masks
def loadFilter():
    filter = cv.imread(filter_path, cv.IMREAD_UNCHANGED)

    filter_gray = cv.cvtColor(filter, cv.COLOR_BGR2GRAY)
    ori_filter_h, ori_filter_w, filter_channels = filter.shape

    # filter mask
    # cv.THRESH_BINARY_INV for png transparent background, cv.THRESH_BINARY for white background
    ret, mask = cv.threshold(filter_gray, 10, 255, cv.THRESH_BINARY_INV)

    mask_inv = cv.bitwise_not(mask)
    
    # mask = cv.threshold(filter[:, :, 2], 0, 255, cv.THRESH_BINARY)[1]

    # trans_mask = filter[:, :, 2] == 255

    # cv.imshow("trans", filter[trans_mask])

    cv.imshow("gray", filter_gray)
    cv.imshow("mask", mask)
    cv.imshow("inv", mask_inv)

    return filter, ori_filter_h, ori_filter_w, mask, mask_inv

# apply filter to face region
def filters(frame, faces, filter, ori_filter_h, ori_filter_w, mask, mask_inv):

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

        # get center of face, deducted by half of filter width
        filter_x1 = face_x2 - int(face_w / 2) - int(filter_width / 2)
        filter_x2 = filter_x1 + filter_width
        # face's upper left -> filter's upper left, adjustable factor
        filter_y1 = face_y1 - int(face_h * 1.26)
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
