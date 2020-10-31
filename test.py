#this test file contains all the test cases I ran to check whether the program is running well or not 
import cv2
import numpy as np
from PIL import Image

im_cv = cv2.imread('input/group of babies/1.jpg')

cv2.imwrite('input/group of babies/1.jp', im_cv)
#print("test")
