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
    cv2.drawContours(img, [board_contour], 0, (0,255,0), 3)
    cv2.imwrite("result.jpg", img)

if __name__ == "__main__":
    main("sudoku.jpg")