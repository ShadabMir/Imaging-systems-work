import cv2 as cv
import numpy as np
from PreprocessingMethods import *
def draw_minimum_area_rect(minarea,image):
    box = cv.boxPoints(minarea)
    box1 = np.int32(box)
    check_image = cv.drawContours(image, [box1], 0, (0, 255, 255), 2)
    cv.imshow("New_image", check_image)
    return box

def getSkewAngle(cvImage,upperArea,LowerArea,lowest_point_rect = 0):

    resized_image = cv.resize(cvImage,(500,500),interpolation=cv.INTER_LINEAR)
    threshold = preprocess_gray_nlmeans_threshold(resized_image)

    contours, hierarchy = cv.findContours(
        threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    cv.imshow('thresh',threshold)

    print(len(contours))
    target_contour = []
    for c in contours:
        rect = cv.boundingRect(c)
        x, y, w, h = rect

        if ((w * h) < (upperArea) ) & ((w * h) > (LowerArea)):
                target_contour.append(c)
                rectangle2 = cv.rectangle(threshold, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(len(target_contour))
    cv.imshow('Rectangle2', rectangle2)
    cv.waitKey(0)
    minAreaRect_ = cv.minAreaRect(target_contour[0])
    width,height = minAreaRect_[1]
    print(width)
    print(height)
    a,b,c,d = cv.boxPoints(minAreaRect_)
    mapped_points = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    input = np.float32([[(a)],[b],[c],[d]])
    print(len(input))
    print(len(mapped_points))
    M = cv.getPerspectiveTransform(input,mapped_points)
    rotated_im = cv.warpPerspective(resized_image,M,(np.int32(width),np.int32(height)),cv.INTER_LINEAR)
    return rotated_im