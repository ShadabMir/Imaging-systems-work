
import cv2 as cv
import numpy as np
import pytesseract
from ShapeDetection  import determine_shape_type
from DeskewingMethods import getSkewAngle
from PreprocessingMethods import *

def rotate_image2(image):
    print('d')
    rotate_image_yes = getSkewAngle(image,425,419)

    return rotate_image_yes
def draw_boxes_OCR_output(image,height):
    boxes = pytesseract.image_to_boxes(image,
                                       config='--psm 10 --oem 3-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM0987654321',
                                       timeout=2000)
    print(boxes)
    for b in boxes.splitlines():
        b = b.split(" ")
        cv.rectangle(
            image,
            (int(b[1]), height - int(b[2])),
            (int(b[3]), height - int(b[4])),
            (230, 255, 0),
            1,
        )

        cv.imshow('Boxes', image)
        cv.imwrite('Image2.jpeg',image.copy())

compImage = cv.imread('background21.png')
rotated_image = rotate_image2(compImage)
rotated_image2 = preprocess_gray_nlmeans_threshold(rotated_image)
print(rotated_image2)
cv.imshow('Rotated',rotated_image2)

h,w = rotated_image2.shape[:2]
configs ='--psm 10 --oem 3-c tessedit_char_whitelist = QWERTYUIOPASDFGHJKLZXCVBNM0987654321'
"""print(pytesseract.image_to_string(rotated_image2))"""
output = pytesseract.image_to_string(rotated_image2,config=configs)
output_array = list(output)
print(output_array)
for items in output_array:
    if items.islower():
        y = 0
    else:
        print('The recognized character is ',items)

draw_boxes_OCR_output(rotated_image2,h)
cv.waitKey(0)