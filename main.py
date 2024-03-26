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
from undistortion import undistortion

imgL = cv2.imread("./photos/14-28-55_left.png", 0)
imgL = undistortion(imgL)
imgR = cv2.imread("./photos/14-28-55_right.png", 0)
imgR = undistortion(imgR)

# cv2.imshow('imgL', imgL)
# cv2.imshow('imgR', imgR)

print(imgR.shape)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=11)
disparity = stereo.compute(imgL, imgR)

print(disparity)
print(disparity.min(), disparity.max(), type(disparity.max()))

norm_image = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imwrite("./photos/disparity_normalized.png", norm_image)
# cv2.imshow('disparity_normalized', norm_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
