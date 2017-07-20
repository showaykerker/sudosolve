"""
__name__   = grid.py
__author__ = Yash Patel
"""

import numpy as np
import cv2

def solve_puzzle():
    pass

def create_from_image():
    pass

def transform_image(img):
    size = 5
    smooth_kernel = np.ones((size,size),np.float32)/(size ** 2)
    smoothed = cv2.filter2D(img,-1,smooth_kernel)
    thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(thresh,100,200)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    
    for line in lines:
        rho, theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow("edges", edges)
    cv2.waitKey(0)

    cv2.imshow("sudoku", img)
    cv2.waitKey(0)

def main(fn):
    image = cv2.imread(fn, 0)
    transform_image(image)

    #cv2.imshow("sudoku", image)
    #cv2.waitKey(0)

if __name__ == "__main__":
    filename = "sudoku.jpg"
    main(filename)