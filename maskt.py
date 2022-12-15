import os
import cv2
import numpy as np
src = cv2.imread('fish.png', 0)
ret, mask = cv2.threshold(src, 255, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('src', src)
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()