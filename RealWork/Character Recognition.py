
import cv2 as cv
import numpy as np
import pytesseract
from ShapeDetection  import determine_shape_type
def preprocess_gray_smooth_threshold(image):
    grayScaleImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    smoothImage = cv.bilateralFilter(grayScaleImage,5,15,15,cv.BORDER_WRAP)
    thresholdedImage = cv.adaptiveThreshold(smoothImage,235,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,3)

    return thresholdedImage

def preprocess_gray_nlmeans_threshold(image):
    grayscaleImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    denoisedImage = cv.fastNlMeansDenoising(grayscaleImage,None,10,7,21)
    thresholdedImage = cv.adaptiveThreshold(denoisedImage,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,3)

    return thresholdedImage

def draw_minimum_area_rect(minarea,image):
    box = cv.boxPoints(minarea)
    box1 = np.int32(box)
    check_image = cv.drawContours(image, [box1], 0, (0, 255, 255), 2)
    cv.imshow("New_image", check_image)
    return box

def centreOfRotation(contour_box):
    y_values = []
    x_values = []
    for bo in contour_box:
        y_values.append([bo[1], bo[0]])
        x_values.append(bo[1])

    x_value = max(x_values)
    for item in y_values:
        if item[0] == x_value:
            a_value = item[1]
def getSkewAngle(cvImage,upperArea,LowerArea,lowest_point_rect = 0):

    resized_image = cv.resize(cvImage,(500,500),interpolation=cv.INTER_LINEAR)
    threshold = preprocess_gray_nlmeans_threshold(resized_image)

    contours, hierarchy = cv.findContours(
        threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    target_contour = []
    for c in contours:
        rect = cv.boundingRect(c)
        x, y, w, h = rect


        if ((w * h) < (upperArea) ) & ((w * h) > (LowerArea)):
                target_contour.append(c)
                rectangle2 = cv.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('Rectangle2', rectangle2)
    minAreaRect_ = cv.minAreaRect(target_contour[0])
    angle = minAreaRect_[-1]
    print(len(minAreaRect_))
    box = draw_minimum_area_rect(minAreaRect_,resized_image)
    centre_of_rotation = centreOfRotation(box)
    if angle < -45:
        angle = 90 + angle
    return -1.0 * (90-angle),target_contour[0],centre_of_rotation


def rotate_image(cvImage,target,centre, angle: float):
    print('e')
    newImage = cvImage.copy()
    newImage = cv.resize(newImage.copy(),(500,500),interpolation=cv.INTER_LINEAR)
    (h, w) = (newImage.shape[0],newImage.shape[1])
    M = cv.getRotationMatrix2D(centre, angle, 0.8)
    newImage = cv.warpAffine(newImage, M, (h,w))
    cv.imshow('new image test',newImage)
    return newImage

def rotate_image2(image):
    print('d')
    angle,targetContour,centreOfRotation = getSkewAngle(image,425,419)
    newImage = rotate_image(image,targetContour,centreOfRotation,angle)
    return newImage
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
            (0, 255, 0),
            2,
        )
    cv.imshow('Boxes', image)


compImage = cv.imread('Red_Cross_White_9.jpeg')
rotated_image = rotate_image2(compImage)
rotated_image2 = preprocess_gray_smooth_threshold(rotated_image)
cv.imshow('Rotated',rotated_image2)

print(pytesseract.image_to_string(rotated_image2,config='--psm 10 --oem 3-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM0987654321',timeout=2000))
h, w = rotated_image2.shape[:2]
draw_boxes_OCR_output(rotated_image,h)
cv.waitKey(0)