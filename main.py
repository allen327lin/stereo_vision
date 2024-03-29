"""
MIT License

Copyright (c) 2024 allen327lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import cv2
import argparse
from undistortion import undistortion
from show_histogram import show_histogram

imgL = None
imgR = None
numDisparities = 16
blockSize = 11
anti_back_threshold = 220

def compute_disparity():
    global imgL
    global imgR
    global numDisparities
    global blockSize
    global anti_back_threshold

    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)

    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    anti_back = np.where(disparity_normalized > anti_back_threshold, imgL, 0)

    cv2.imshow('disparity_normalized', disparity_normalized)
    cv2.imwrite("./photos/disparity_normalized.png", disparity_normalized)
    cv2.imwrite("./photos/anti_back.png", anti_back)

    show_histogram(disparity, "disparity")
    show_histogram(disparity_normalized, "disparity_normalized")

    return [disparity, disparity_normalized]

def update_numDisparities(val):
    global numDisparities
    # val * 16 = numDisparities
    numDisparities = val * 16
    compute_disparity()

def update_blockSize(val):
    global blockSize
    # val * 2 + 1 = blockSize
    blockSize = val * 2 + 1
    compute_disparity()

def update_anti_back_threshold(val):
    global anti_back_threshold
    anti_back_threshold = val
    compute_disparity()


def main():
    global imgL
    global imgR

    imgL_path = "./photos/44_left_downsized_0.2.png"
    imgR_path = "./photos/44_right_downsized_0.2.png"

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--imgL_path", type=str, help="Left photo path")
    parser.add_argument("-r", "--imgR_path", type=str, help="Right photo path")
    args = parser.parse_args()
    if args.imgL_path is not None:
        imgL_path = args.imgL_path
    if args.imgR_path is not None:
        imgR_path = args.imgR_path

    imgL = cv2.imread(imgL_path, 0)
    imgR = cv2.imread(imgR_path, 0)
    imgL = undistortion(imgL)
    imgR = undistortion(imgR)

    cv2.namedWindow('disparity_normalized', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disparity_normalized', 1280, 820)

    disparity, disparity_normalized = compute_disparity()

    cv2.imshow('disparity_normalized', disparity_normalized)

    cv2.createTrackbar('numDisparities', 'disparity_normalized', 1, 100, update_numDisparities)
    cv2.createTrackbar('blockSize', 'disparity_normalized', 5, 100, update_blockSize)
    cv2.createTrackbar('anti_back_threshold', 'disparity_normalized', 220, 255, update_anti_back_threshold)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()