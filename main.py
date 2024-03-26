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
