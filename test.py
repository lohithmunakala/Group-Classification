import cv2
import numpy as np
from PIL import Image

im_cv = cv2.imread('input/group of babies/1.jpg')

cv2.imwrite('input/group of babies/1.jp', im_cv)
