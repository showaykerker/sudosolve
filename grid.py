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
    kernel = np.ones((size,size),np.float32)/(size ** 2)
    print(kernel)
    smoothed = cv2.filter2D(img,-1,kernel)
    thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY, 11, 2)
    connectivity = 4
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    num_labels = output[0]
    labels = output[1]
    sizes = np.array([np.sum(labels == label) for label in range(num_labels)])
    # label_with_sizes = dict(zip(range(num_labels), sizes))

    sorted_size_indices = np.argsort(sizes)
    border_index = labels[sorted_size_indices[9]]
    filtered_img = (labels == border_index).astype(np.float32)

    cv2.imshow("sudoku", filtered_img)
    cv2.waitKey(0)

def main(fn):
    image = cv2.imread(fn, 0)
    transform_image(image)

    #cv2.imshow("sudoku", image)
    #cv2.waitKey(0)

if __name__ == "__main__":
    filename = "sudoku.jpg"
    main(filename)