"""
__name__   = grid.py
__author__ = Yash Patel
"""

from operator import itemgetter
import numpy as np
import cv2

def extrema_pts(comp_vals, fn, xs, ys):
    """
    Description: Given comparison array and function, finds the extremum and
    the corresponding x/y values. Returns as array of pairs

    Params:
        comp_vals (num arr): Array of values to be used to determine what the
            extremum of interest is
        fn (fun: num arr -> num): Function to pick out extremum (max/min)
        xs (num arr): x values from which extremum is returned
        ys (num arr): y values from which extremum is returned
    """
    extremum = comp_vals.index(fn(comp_vals))
    x1, x2 = xs[extremum]
    y1, y2 = ys[extremum]
    return [[x1, y1],[x2, y2]]

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) 

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]

def solve_puzzle():
    pass

def create_from_image():
    pass

def transform_image(img):
    size = 5
    smooth_kernel = np.ones((size,size),np.float32)/(size ** 2)
    smoothed = cv2.filter2D(img,-1,smooth_kernel)
    thresh = cv2.adaptiveThreshold(smoothed, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(thresh,100,200)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)

    horizontal_xs = []
    horizontal_ys = []
    vertical_xs = []
    vertical_ys = []

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
        
        hor_dist  = abs(x2 - x1)   # distance left/right btw endpoints
        vert_dist = abs(y2 - y1)   # distance up/down btw endpoints
        is_horizontal = False # denotes whether or not the line is horizontal
        if hor_dist > vert_dist:
            is_horizontal = True

        if is_horizontal: 
            horizontal_xs.append((x1,x2))
            horizontal_ys.append((y1,y2))
        else:
            vertical_xs.append((x1,x2))
            vertical_ys.append((y1,y2))

    #------------------------------------------------------------------------- #
    # Crops out just the sudoku board from the image given the Hough transform #
    #------------------------------------------------------------------------- #
    top    = extrema_pts(horizontal_ys, min, horizontal_xs, horizontal_ys)
    bottom = extrema_pts(horizontal_ys, max, horizontal_xs, horizontal_ys)
    left   = extrema_pts(vertical_xs, min, vertical_xs, vertical_ys)
    right  = extrema_pts(vertical_xs, max, vertical_xs, vertical_ys)

    #------------------------------------------------------------------------- #
    # Crops the image   #
    #------------------------------------------------------------------------- #

    top_left     = line_intersection(top, left)
    top_right    = line_intersection(top, right)
    bottom_left  = line_intersection(bottom, left)
    bottom_right = line_intersection(bottom, right)
    
    pts1 = np.float32([top_left,top_right,bottom_left,bottom_right])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M   = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (300,300))

    #------------------------------------------------------------------------- #
    cv2.imshow("sudoku", dst)
    cv2.waitKey(0)

def main(fn):
    image = cv2.imread(fn, 0)
    transform_image(image)

if __name__ == "__main__":
    filename = "sudoku.jpg"
    main(filename)