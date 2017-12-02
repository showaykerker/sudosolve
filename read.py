"""
__author__ = Yash Patel
__name__   = read.py
__description__ = Given an image, produces/determines the corresponding Sudoku board 
that is present in the image
"""

import cv2
import numpy as np

def extract_board(img):
    pass

def convert_form(feature):
    # order has to be:
    # [top_left, top_right, bottom_right, bottom_left]
    sums = [sum(pt[0]) for pt in feature]
    unused = {0,1,2,3}

    top_left_ind = np.argmin(sums)
    bottom_right_ind = np.argmax(sums)

    top_left = feature[top_left_ind]
    
    top_left_dist = [(abs(top_left[0][1] - pt[0][1]), i) 
        for i, pt in enumerate(feature)]
    top_right_ind = sorted(top_left_dist)[1][1]

    unused.remove(top_left_ind)
    unused.remove(bottom_right_ind)
    unused.remove(top_right_ind)
    bottom_left_ind = list(unused)[0]
    return np.array([feature[top_left_ind],feature[top_right_ind],
            feature[bottom_right_ind],feature[bottom_left_ind]])

def main(fn):
    img    = cv2.imread(fn)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE)

    max_rectangle = None
    areas = []

    AREA_THRESHOLD = 100
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > AREA_THRESHOLD:
            perim  = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, .02 * perim, True)
            if len(approx) == 4:
                areas.append((area, i))
    
    areas.sort(reverse=True)
    board_contour = contours[areas[1][1]]
    board_perim  = cv2.arcLength(board_contour, True)
    best_rect = cv2.approxPolyDP(board_contour, .02 * board_perim, True)
    form_rect = convert_form(best_rect)

    cv2.drawContours(img, [form_rect], 0, (0,255,0), 3)
    cv2.imwrite("result.jpg", img)

if __name__ == "__main__":
    main("sudoku.jpg")