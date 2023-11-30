import cv2 as cv
import numpy as np
def preprocess_gray_smooth_threshold(image):
    grayScaleImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    smoothImage = cv.bilateralFilter(grayScaleImage,5,15,15,cv.BORDER_WRAP)
    _,thresholdedImage = cv.threshold(smoothImage,170,255,cv.THRESH_BINARY_INV)

    return thresholdedImage

def preprocess_gray_nlmeans_threshold(image):
    grayscaleImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    denoisedImage = cv.fastNlMeansDenoising(grayscaleImage,None,10,7,21)
    bilateralimage = cv.bilateralFilter(grayscaleImage,5,15,15,cv.BORDER_WRAP)
    _,thresholdedImage = cv.threshold(denoisedImage,125,255,cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(1,1))
    morphed = cv.morphologyEx(thresholdedImage,cv.MORPH_CLOSE,kernel,iterations=1)
    return morphed