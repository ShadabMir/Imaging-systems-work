from rembg import remove
import cv2 as cv
import easygui as eg
from PIL import Image

input_image = cv.imread("Red_Cross_White_9.jpeg")
output_image = remove(input_image)
cv.imwrite("background21.png",output_image)
