# import statements
import cv2 as cv

# initial filter parameters
filter_path = 'filters/witch.png'
filter = cv.imread(filter_path)
filter_gray = cv.cvtColor(filter, cv.COLOR_BGR2GRAY)
ori_filter_h, ori_filter_w, filter_channels = filter.shape

# filter mask
# cv.THRESH_BINARY_INV for png transparent background, cv.THRESH_BINARY for white background
ret, mask = cv.threshold(filter_gray, 10, 255, cv.THRESH_BINARY_INV)
mask_inv = cv.bitwise_not(mask)

# apply filter to face region
def filters(faces):
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
        filter_width = int(1.6 * face_w)
        filter_height = int(filter_width * (ori_filter_h / ori_filter_w))

    print("note!")
