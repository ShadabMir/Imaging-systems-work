import cv2 as cv
import numpy as np
import pytesseract
from ShapeDetection import determine_shape_type
def geo_location(orgimage,image,altitude,GSD,bearingAngle,drone_location,maximum_area=100000000000):
    imageWidth = 500
    pixel_per_cm = GSD
    cv.imshow('Image3',image)
    contours,hierarchy = cv.findContours(image,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    print(len(contours))
    contours2 = contours[0:10]
    contour_centre = []


    for c in contours:
        b = cv.boundingRect(c)
        x,y,w,h = b
        if ((w*h) > 3150) & ((w*h)<3220):

            if determine_shape_type(c) == "UNKNOWN":
                print('x')
            else:
                contour_centre.append(x+(w/2))
                contour_centre.append((y+(h/2)))

        else:
            print('wwe')
    image_centre = (250,250)
    print(contour_centre,'Contour centre')
    L = np.sqrt((contour_centre[0]-image_centre[0])**2 + (contour_centre[1]-image_centre[1])**2)
    R = 6300000
    lat1,lon1 = drone_location
    lat2 = np.arcsin(((np.sin(lat1)*(np.sin(L/R))) + (np.cos(lat1))*np.sin(L/R)*np.cos(bearingAngle)))
    lon2 = lon1 + np.arctan((np.sin(bearingAngle)*np.sin(L/R)*np.cos(lat1))/((np.cos(L/R)-np.sin(lat1))*np.sin(lat2)))
    return lat2,lon2


def preprocess_gray_blur_denoise_thresh_morph(image):

    gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    blur_image = cv.bilateralFilter(gray_image,5,20,20,cv.BORDER_DEFAULT)
    image_threshold = cv.adaptiveThreshold(blur_image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,3)
    image_denoise = cv.fastNlMeansDenoising(image_threshold,None,10,7,21)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    image_morph = cv.morphologyEx(image_threshold,cv.MORPH_DILATE,kernel,cv.BORDER_DEFAULT)
    return image_morph

def resize_image(image,h,w):
    resizedImage = cv.resize(image.copy(),(h,w),cv.BORDER_DEFAULT)

test_image = cv.imread('RealWork/Red_Cross_White_9.jpeg')
resized_image = resize_image(test_image,500,500)
preprocessed_image = preprocess_gray_blur_denoise_thresh_morph(resized_image)
latitude_object,longitude_object = geo_location(resized_image,preprocessed_image,100,2650,(np.pi/3),(100,150))
cv.waitKey(0)
