import cv2
import numpy as np

im1 = cv2.imread('../data/mytest/20211230_094238.png')
im2 = cv2.imread('../data/mytest/20211230_094347.png')

src_points = np.array([[1832.0, 755.0], [2091.0, 755.0], [2091.0, 1060.0], [1832.0, 1060.0]])
dst_points = np.array([[1931.0, 737.0], [2187.0, 737.0], [2187.0, 1043.0], [1931.0, 1043.0]])

H, _ = cv2.findHomography(src_points, dst_points)

h, w = im2.shape[:2]

im2_warp = cv2.warpPerspective(im2, H, (w, h))

# cv2.imshow('im2', im2_warp)
cv2.imwrite('../data/mytest/im2.png', im2_warp, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


