import cv2 as cv
import numpy as np
import pytesseract
import PIL
from ShapeDetection import determine_shape_type

tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = tesseract_path
"""cam = cv.VideoCapture(0)
key_in = cv.waitKey(1)
cap_count = 0while cam.isOpened():
    key_in_last = key_in
    key_in = cv.waitKey(1)
    ret, image = cam.read()
    if not ret:
        print("Cant recieve frame")
        break
    elif key_in == ord("p"):
        print("Quit pressed,Quitting")
        break

    cv.imshow("Live feed", image)
    if key_in == ord(" "):
        cap_count += 1
        cv.imshow("Image", image)


cam.release()"""


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
        if ((w*h) > 419) & ((w*h)<425):
            rectImage = cv.rectangle(image, (x,y), (x+w,y+h), color=(200,230,255), thickness=2)
            cv.imshow('Rect',rectImage)
            print("a")
            if determine_shape_type(c) == "UNKNOWN":
                print('x')
            else:
                contour_centre.append(x+(w/2))
                contour_centre.append((y+(h/2)))

        else:
           y=0
    image_centre = (250,250)
    print(contour_centre,'Contour centre')
    L = np.sqrt((contour_centre[0]-image_centre[0])**2 + (contour_centre[1]-image_centre[1])**2)
    R = 6300000
    lat1,lon1 = drone_location
    lat2 = np.arcsin(((np.sin(lat1)*(np.sin(L/R))) + (np.cos(lat1))*np.sin(L/R)*np.cos(bearingAngle)))
    lon2 = lon1 + np.arctan((np.sin(bearingAngle)*np.sin(L/R)*np.cos(lat1))/((np.cos(L/R)-np.sin(lat1))*np.sin(lat2)))
    return lat2,lon2


image = cv.imread('RealWork/Red_Cross_White_9.jpeg')
resized_image = cv.resize(image.copy(),(500,500),cv.BORDER_DEFAULT)
gray_image = cv.cvtColor(resized_image,cv.COLOR_BGR2GRAY)
blur_image = cv.bilateralFilter(gray_image,5,20,20,cv.BORDER_DEFAULT)
image_threshold = cv.adaptiveThreshold(blur_image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,3)
image_denoise = cv.fastNlMeansDenoising(image_threshold,None,10,7,21)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
image_morph = cv.morphologyEx(image_threshold,cv.MORPH_DILATE,kernel,cv.BORDER_DEFAULT)
cv.imshow('Iamge',image_threshold)
latitude_object,longitude_object = geo_location(resized_image,image_denoise,100,2650,(np.pi/3),(100,150))
print(latitude_object,'Latitude')
print(longitude_object,'Longitude')

cv.waitKey(0)